#include <dlfcn.h>
#include <chrono>
#include <future>
#include <utility>

#include "checker.h"
#include "logger.h"
#include "session.h"

#include <ATen/record_function.h>
#include "acl/acl_rt.h"
#include "acl/acl_tdt.h"
#include "ge/ge_api_types.h"
#include "ge/ge_api.h"
#include "hdc_channel.h"
#include "npu_aoe.h"

namespace {
std::unique_ptr<ge::Session> global_ge_session = nullptr;
void *libge_runner_handle = nullptr;

bool IsGraphNeedLogChannel(const ge::Graph &graph) {
  const static std::string kPrintV2OpType = "PrintV2";
  for (auto &node : graph.GetAllNodes()) {
    ge::AscendString op_type("");
    if ((node.GetType(op_type) == ge::GRAPH_SUCCESS) && (op_type.GetString() == kPrintV2OpType)) {
      return true;
    }
  }
  return false;
}
}  // namespace

namespace tng {

Session &Session::GetInstance() {
  static Session instance;
  return instance;
}

Status Session::Initialize(const std::map<std::string, std::string> &options) {
  if (initialized_) {
    return status_;
  }
  std::lock_guard<std::mutex> const lock(mu_);
  if (initialized_) {
    return status_;
  }

  std::map<ge::AscendString, ge::AscendString> ge_options;
  TNG_LOG(INFO) << "Initializing GE with options:";
  for (const auto &option : options) {
    TNG_LOG(INFO) << "  " << option.first << ": " << option.second;
    if (option.first == "ge_run_with_torch_npu") {
      run_with_torch_npu_ = option.second == "1";
      continue;
    }
    ge_options[option.first.c_str()] = option.second.c_str();
  }

  auto iter = ge_options.find(ge::AscendString(ge::OPTION_EXEC_DEVICE_ID));
  TNG_ASSERT(iter != ge_options.end(), "Device id is not specified when initializing GE");
  // the context will switch after ge Session created
  // set device index in option in order to keep original context
  device_index_ = static_cast<int32_t>(std::atoi(iter->second.GetString()));
  TNG_ASSERT(device_index_ >= 0, "device_index_ = %d, assert device_index_ >= 0 failed!", device_index_);

  if (ge::GEInitialize(ge_options) != ge::SUCCESS) {
    status_ = Status::Error("Failed to initialize GE %s", compat::GeErrorStatus().GetErrorMessage());
  } else {
    (void)ge_options.emplace(ge::AscendString("ge.session_device_id"), iter->second);
    global_ge_session = std::make_unique<ge::Session>(ge_options);
    if (global_ge_session == nullptr) {
      status_ = Status::Error("Failed to create GE session");
    }
  }

  auto ret = aclrtSetDevice(device_index_);
  TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL set device id failed, return %d", ret);

  libge_runner_handle = dlopen("libge_runner.so", RTLD_NOW);
  TNG_ASSERT_NOTNULL(libge_runner_handle, "libge_runner.so dlopen failed, %s", dlerror());
  fast_load_graph_ = reinterpret_cast<GeSessionLoadGraphFunc *>(dlsym(libge_runner_handle, "GeSessionLoadGraph"));
  fast_execute_graph_async_ =
      reinterpret_cast<GeFastExecuteGraphFunc *>(dlsym(libge_runner_handle, "GeSessionExecuteGraphWithStreamAsync"));
  TNG_LOG(DEBUG) << "In current cann version"
                 << ", FastLoadGraph api is " << (IsFastLoadGraphSupported() ? "supported" : "unsupported")
                 << ", FastExecuteGraph api is " << (IsFastExecuteGraphSupported() ? "supported" : "unsupported");
  initialized_ = true;
  return status_;
}

Status Session::EnsureInitialized() {
  if (initialized_) {
    return status_;
  }
  return Status::Error("Session is not initialized");
}

Status Session::Finalize() {
  if (!initialized_) {
    return Status::Success();
  }

  TNG_LOG(DEBUG) << "Start to synchronize device in Finalize.";
  auto sync_ret = aclrtSynchronizeDevice();
  if (sync_ret != ACL_ERROR_NONE) {
    TNG_LOG(ERROR) << "ACL synchronize device failed in Finalize, return " << sync_ret;
  } else {
    TNG_LOG(DEBUG) << "ACL synchronize device success in Finalize.";
  }

  global_ge_session.reset(nullptr);
  StopStdoutChannel(device_index_);  // Stopped after all graph run finished

  fast_load_graph_ = nullptr;
  fast_execute_graph_async_ = nullptr;
  if (libge_runner_handle) {
    dlclose(libge_runner_handle);
    libge_runner_handle = nullptr;
  }
  if (!run_with_torch_npu_) {
    TNG_LOG(DEBUG) << "Start to GEFinalize.";
    TNG_ASSERT_GE_OK(ge::GEFinalize());
  }

  auto result = Status::Success();
  if (auto_tune_init_) {
    result = NpuAoe::GetInstance().AoeTuningFinalize();
  }

  aclrtContext detect_context = aclrtContext();
  auto ctx_ret = aclrtGetCurrentContext(&detect_context);
  auto ctx_ptr = (ctx_ret == ACL_ERROR_NONE) ? detect_context : nullptr;

  initialized_ = false;

  TNG_LOG(DEBUG) << "After torchair finalize, got context pointer: " << ctx_ptr
                 << ", and the initialized flag is set to " << initialized_;

  return result;
}

Status Session::AddGraph(uint32_t id, const ge::Graph &graph,
                         const std::map<ge::AscendString, ge::AscendString> &options) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());

  if (IsGraphNeedLogChannel(graph)) {
    TNG_RETURN_IF_ERROR(StartStdoutChannel(device_index_));
  }

  TNG_ASSERT_GE_OK(global_ge_session->AddGraph(id, graph, options));
  TNG_LOG(INFO) << "Success to add graph, graph id :" << id;

  return Status::Success();
}

Status Session::CompileGraph(uint32_t id, std::shared_ptr<ge::CompiledGraphSummary> &summary) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());

  std::future<Status> future = std::async(std::launch::async, [&]() {
    auto start = std::chrono::high_resolution_clock::now();
    TNG_ASSERT_GE_OK(global_ge_session->CompileGraph(id));
    auto end = std::chrono::high_resolution_clock::now();
    auto warning_msg = tng::LogLevelEnable(tng::LogLevel::WARNING) ? ge::GEGetWarningMsgV2().GetString() : nullptr;
    if (warning_msg != nullptr && strlen(warning_msg) != 0) {
      TNG_LOG(WARNING) << "During Compile Graph, a warn message occurred. Please refer to the details：" << warning_msg;
    }
    TNG_LOG(EVENT) << "Compile Graph " << id << " consume: "
                   << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count()
                   << " ms.";
    if (summary == nullptr) {
      summary = global_ge_session->GetCompiledGraphSummary(id);
      TNG_ASSERT_NOTNULL(summary, "Failed get compiled summary of graph %d", id);
    }
    return Status::Success();
  });

  return future.get();
}

Status Session::AutoTuneGraph(const ge::Graph &graph, const std::map<ge::AscendString, ge::AscendString> &options,
                              const std::vector<ge::Tensor> &example_inputs, void *stream) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());

  TNG_LOG(INFO) << "Auto tuning graph";
  TNG_RETURN_IF_ERROR(NpuAoe::GetInstance().AoeTuningInitialize(options.at("work_path"), options.at("aoe_mode")));
  auto_tune_init_ = true;
  {
    ge::Graph clone_graph("aoe_aopied_graph");
    TNG_ASSERT_GE_OK(clone_graph.CopyFrom(graph));

    (void)NpuAoe::GetInstance().RunAoeTuning(clone_graph, options, example_inputs, stream, global_ge_session.get());
  }
  return Status::Success();
}

Status Session::RemoveGraph(uint32_t id) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());

  TNG_ASSERT_GE_OK(global_ge_session->RemoveGraph(id));
  TNG_LOG(INFO) << "Success to remove graph, graph id :" << id;

  return Status::Success();
}

Status Session::RegisterExternalAllocator(const void *const stream, std::shared_ptr<ge::Allocator> allocator) {
  TNG_ASSERT_GE_OK(global_ge_session->RegisterExternalAllocator(stream, allocator));
  return Status::Success();
}

Status Session::RunGraph(uint32_t id, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs,
                         void *stream) {
  RECORD_FUNCTION("RunGraph", {});
  TNG_RETURN_IF_ERROR(EnsureInitialized());
  TNG_LOG(DEBUG) << "Start to session run graph " << id;

  if (stream == nullptr) {
    TNG_ASSERT_GE_OK(global_ge_session->RunGraph(id, inputs, outputs));
  } else {
    TNG_ASSERT_GE_OK(global_ge_session->RunGraphWithStreamAsync(id, stream, inputs, outputs));
  }

  return Status::Success();
}

Status Session::SetGraphConstMemoryBase(uint32_t id, const void *const memory, size_t size) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());
  TNG_ASSERT_GE_OK(global_ge_session->SetGraphConstMemoryBase(id, memory, size));
  return Status::Success();
}

Status Session::SetGraphFixedFeatureMemoryBase(uint32_t id, const void *const memory, size_t size) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());
  TNG_ASSERT_GE_OK(global_ge_session->SetGraphFixedFeatureMemoryBase(id, memory, size));
  return Status::Success();
}

Status Session::UpdateGraphFeatureMemoryBase(uint32_t id, const void *const memory, size_t size) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());
  TNG_ASSERT_GE_OK(global_ge_session->UpdateGraphFeatureMemoryBase(id, memory, size));
  return Status::Success();
}

Status Session::UpdateGraphRefreshableFeatureMemoryBase(uint32_t id, const void *const memory, size_t size) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());
  TNG_ASSERT_GE_OK(global_ge_session->UpdateGraphRefreshableFeatureMemoryBase(id, memory, size));
  return Status::Success();
}

Status Session::FastLoadGraph(uint32_t graph_id, const std::map<ge::AscendString, ge::AscendString> &option,
                              void *stream) {
  RECORD_FUNCTION("LoadGraph", {});
  TNG_RETURN_IF_ERROR(EnsureInitialized());
  TNG_ASSERT_NOTNULL(fast_load_graph_, "FastLoadGraph is unsupported, please dont use it in current cann version.");
  TNG_LOG(DEBUG) << "Start to session load graph " << graph_id <<", load options:";
  for (const auto &opt : option) {
    TNG_LOG(DEBUG) << "  " << opt.first.GetString() << ": " << opt.second.GetString();
  }

  TNG_ASSERT_GE_OK(fast_load_graph_(*global_ge_session, graph_id, option, stream));
  TNG_LOG(INFO) << "Success to load graph, graph id :" << graph_id;
  return Status::Success();
}

Status Session::FastExecuteGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs,
                                 std::vector<gert::Tensor> &outputs, void *stream) {
  RECORD_FUNCTION("ExecuteGraph", {});
  TNG_RETURN_IF_ERROR(EnsureInitialized());
  TNG_ASSERT_NOTNULL(fast_execute_graph_async_,
                     "FastExecuteGraph is unsupported, please dont use it in current cann version.");
  TNG_LOG(DEBUG) << "Start to session execute graph " << graph_id;

  TNG_ASSERT_GE_OK(fast_execute_graph_async_(*global_ge_session, graph_id, stream, inputs, outputs));
  return Status::Success();
}
}  // namespace tng
