#include <future>
#include <chrono>

#include "checker.h"
#include "logger.h"
#include "session.h"
#include "utils.h"

#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "acl/acl_rt.h"
#include "npu_aoe.h"

namespace {
std::unique_ptr<ge::Session> global_ge_session = nullptr;
}  // namespace

namespace tng {
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

  if (ge::GEInitialize(ge_options) != ge::SUCCESS) {
    status_ = Status::Error("Failed to initialize GE %s", compat::GeErrorStatus().GetErrorMessage());
  } else {
    (void)ge_options.emplace(ge::AscendString("ge.session_device_id"), iter->second);
    global_ge_session = std::make_unique<ge::Session>(ge_options);
    if (global_ge_session == nullptr) {
      status_ = Status::Error("Failed to create GE session");
    }
  }
  // the context will switch after ge Session created
  // set device index in option in order to keep original context
  device_index_ = static_cast<int32_t>(std::atoi(iter->second.GetString()));
  TNG_ASSERT(device_index_ >= 0, "device_index_ = %d, assert device_index_ >= 0 failed!", device_index_);
  auto ret = aclrtSetDevice(device_index_);
  TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL set device id failed, return %d", ret);

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
  // reset device
  auto ret = aclrtResetDevice(device_index_);
  if (ret != ACL_ERROR_NONE) {
    TNG_LOG(WARNING) << "ACL reset device index " << device_index_ << " failed, returned " << ret;
  }

  global_ge_session.reset(nullptr);
  if (!run_with_torch_npu_) {
    TNG_ASSERT_GE_OK(ge::GEFinalize());
  }

  if (auto_tune_init_) {
    return NpuAoe::GetInstance().AoeTuningFinalize();
  }

  return Status::Success();
}

Status Session::AddGraph(uint32_t id, const ge::Graph &graph,
                         const std::map<ge::AscendString, ge::AscendString> &options) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());

  TNG_ASSERT_GE_OK(global_ge_session->AddGraph(id, graph, options));

  return Status::Success();
}

Status Session::CompileGraph(uint32_t id, std::shared_ptr<ge::CompiledGraphSummary> *summary) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());

  std::future<Status> future = std::async(std::launch::async, [&]() {
    auto start = std::chrono::high_resolution_clock::now();
    TNG_ASSERT_GE_OK(global_ge_session->CompileGraph(id));
    auto end = std::chrono::high_resolution_clock::now();
    TNG_LOG(EVENT) << "Compile Graph " << id << " consume: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count()
                  << " ms.";
    if (summary != nullptr) {
      *summary = global_ge_session->GetCompiledGraphSummary(id);
      TNG_ASSERT_NOTNULL(*summary, "Failed get compiled summary of graph %d", id);
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

  return Status::Success();
}

Status Session::RegisterExternalAllocator(const void *const stream, std::shared_ptr<ge::Allocator> allocator) {
  TNG_ASSERT_GE_OK(global_ge_session->RegisterExternalAllocator(stream, allocator));
  return Status::Success();
}

Status Session::RunGraph(uint32_t id, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs,
                         void *stream) {
  TNG_RETURN_IF_ERROR(EnsureInitialized());

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
}  // namespace tng