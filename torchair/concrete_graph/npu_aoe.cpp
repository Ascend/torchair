#include "npu_aoe.h"
#include "logger.h"
#include "checker.h"

#include <dlfcn.h>

namespace {
constexpr int32_t aoeSuccessStatus = 0;
constexpr int32_t aoeErrorNonOptimizerGraphStatus = 8;
} // namespace

namespace tng {
NpuAoe &NpuAoe::GetInstance() {
  static NpuAoe instance;
  return instance;
}

Status NpuAoe::RunAoeTuning(const ge::Graph &graph, const std::map<ge::AscendString, ge::AscendString> &options,
                            const std::vector<ge::Tensor> &example_inputs, void *stream, ge::Session *geSession) {
  TNG_LOG(INFO) << "Start to run aoe_tuning";
  SessionKey aoe_session_key = 0UL;
  auto ret = aoe_func_.aoe_create_session(aoe_session_key);
  TNG_ASSERT(ret == aoeSuccessStatus, "Exec aoe create session func failed, error code: %d", ret);

  ret = aoe_func_.aoe_set_gesession(aoe_session_key, geSession);
  TNG_ASSERT(ret == aoeSuccessStatus, "Exec aoe set session func failed, error code: %d", ret);

  // set tuning graph
  ret = aoe_func_.aoe_set_tuninggraph(aoe_session_key, graph);
  TNG_ASSERT(ret == aoeSuccessStatus, "Exec aoe set tuning graph func failed, error code: %d", ret);

  // set tuning inputs
  ret = aoe_func_.aoe_set_tuning_graph_input(aoe_session_key, example_inputs);
  TNG_ASSERT(ret == aoeSuccessStatus, "Exec aoe set tuning inputs func failed, error code: %d", ret);

  // aoe tuning
  std::map<ge::AscendString, ge::AscendString> tuning_options;
  if (options.find(ge::AscendString("ge.aoe_config_file")) != options.cend()) {
    (void)tuning_options.emplace(ge::AscendString("ge.aoe_config_file"), options.at(ge::AscendString("aoe_config_file")));
  }
  ret = aoe_func_.aoe_tuning_graph(aoe_session_key, tuning_options);
  TNG_ASSERT(ret == aoeSuccessStatus || ret == aoeErrorNonOptimizerGraphStatus,
      "Exec aoe set tuning inputs func failed, error code: %d", ret);

  ret = aoe_func_.aoe_destroy_session(aoe_session_key);
  TNG_ASSERT(ret == aoeSuccessStatus, "Exec aoe destroy session func failed, error code: %d", ret);

  TNG_LOG(INFO) << "Success to run aoe_tuning";
  return Status::Success();
}

Status NpuAoe::AoeTuningInitialize(const ge::AscendString &work_path, const ge::AscendString &job_type) {
  TNG_LOG(INFO) << "Start to run aoe initialize";

  handle_ = dlopen("libaoe_tuning.so", RTLD_NOW);
  TNG_ASSERT_NOTNULL(handle_, "libaoe_tuning.so dlopen failed.");
  TNG_RETURN_IF_ERROR(LoadAoeFunc());

  std::map<ge::AscendString, ge::AscendString> global_options;
  (void)global_options.emplace(ge::AscendString("work_path"), work_path);
  (void)global_options.emplace(ge::AscendString("job_type"), job_type);
  auto ret = aoe_func_.aoe_initialize(global_options);
  TNG_ASSERT(ret == aoeSuccessStatus, "Exec aoe initialize func failed, error code: %d", ret);

  TNG_LOG(INFO) << "Run aoe initialize success";
  return Status::Success();
}

Status NpuAoe::LoadAoeFunc() {
  TNG_LOG(INFO) << "Start to load aoe function";

  // aoe init
  aoe_func_.aoe_initialize = reinterpret_cast<AoeInitializeFunc>(dlsym(handle_, "AoeInitialize"));
  TNG_ASSERT(aoe_func_.aoe_initialize != nullptr, "dlsym Aoe Initialize API failed");

  // aoe finalize
  aoe_func_.aoe_finalize = reinterpret_cast<AoeFinalizeFunc>(dlsym(handle_, "AoeFinalize"));
  TNG_ASSERT(aoe_func_.aoe_finalize != nullptr, "dlsym Aoe Finalize API failed");

  // aoe create session
  aoe_func_.aoe_create_session = reinterpret_cast<AoeCreateSessionFunc>(dlsym(handle_, "AoeCreateSession"));
  TNG_ASSERT(aoe_func_.aoe_create_session != nullptr, "dlsym Aoe create session API failed");

  // aoe destroy session
  aoe_func_.aoe_destroy_session = reinterpret_cast<AoeDesstroySessionFunc>(dlsym(handle_, "AoeDestroySession"));
  TNG_ASSERT(aoe_func_.aoe_destroy_session != nullptr, "dlsym Aoe destroy session API failed");

  // aoe set session
  aoe_func_.aoe_set_gesession = reinterpret_cast<AoeSetGeSessionFunc>(dlsym(handle_, "AoeSetGeSession"));
  TNG_ASSERT(aoe_func_.aoe_set_gesession != nullptr, "dlsym Aoe set session API failed");

  // aoe set depend graphs
  aoe_func_.aoe_set_dependgraphs = reinterpret_cast<AoeSetDependGraphFunc>(dlsym(handle_, "AoeSetDependGraphs"));
  TNG_ASSERT(aoe_func_.aoe_set_dependgraphs != nullptr, "dlsym Aoe set depend graphs API failed");

  // aoe set tuning graph
  aoe_func_.aoe_set_tuninggraph = reinterpret_cast<AoeSetTuningGraphFunc>(dlsym(handle_, "AoeSetTuningGraph"));
  TNG_ASSERT(aoe_func_.aoe_set_tuninggraph != nullptr, "dlsym Aoe set tuning graph API failed");

  // aoe tuning
  aoe_func_.aoe_tuning_graph = reinterpret_cast<AoeTuningGraphFunc>(dlsym(handle_, "AoeTuningGraph"));
  TNG_ASSERT(aoe_func_.aoe_tuning_graph != nullptr, "dlsym Aoe tuning graph API failed");

  // aoe set tuning depend graphs inputs
  aoe_func_.aoe_set_depend_graphs_inputs =
    reinterpret_cast<AoeSetDependGraphsInputsFunc>(dlsym(handle_, "AoeSetDependGraphsInputs"));
  TNG_ASSERT(aoe_func_.aoe_set_depend_graphs_inputs != nullptr, "dlsym Aoe set tuning depend graphs inputs API failed");

  // aoe set tuning graph inputs
  aoe_func_.aoe_set_tuning_graph_input =
    reinterpret_cast<AoeSetTuningGraphInputFunc>(dlsym(handle_, "AoeSetTuningGraphInput"));
  TNG_ASSERT(aoe_func_.aoe_set_tuning_graph_input != nullptr, "dlsym Aoe set tuning graph input API failed");

  TNG_LOG(INFO) << "Load aoe function success";
  return Status::Success();
}

Status NpuAoe::AoeTuningFinalize() {
  if (handle_ != nullptr) {
    TNG_LOG(INFO) << "Start to run aoe finalize";

    auto ret = aoe_func_.aoe_finalize();
    TNG_ASSERT(ret == aoeSuccessStatus, "Exec aoe finalize func failed");

    TNG_LOG(INFO) << "Run aoe finalize success";
  }

  return Status::Success();
}

NpuAoe::~NpuAoe() {
  if (handle_ != nullptr) {
    TNG_LOG(INFO) << "Close handle";
    (void)dlclose(handle_);
    handle_ = nullptr;
  }
}

} // namespace tng