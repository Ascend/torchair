#ifndef TORCHAIR_TORCHAIR_INCLUDE_NPU_AOE_H
#define TORCHAIR_TORCHAIR_INCLUDE_NPU_AOE_H

#include <map>
#include "graph/graph.h"
#include "graph/ascend_string.h"
#include "ge/ge_api.h"
#include "tng_status.h"

namespace tng {
using SessionKey = uint64_t;
using AoeStatus = int32_t;
using AoeInitializeFunc = AoeStatus (*)(const std::map<ge::AscendString, ge::AscendString> &);
using AoeFinalizeFunc = AoeStatus (*)();
using AoeCreateSessionFunc = AoeStatus (*)(SessionKey &);
using AoeDesstroySessionFunc = AoeStatus (*)(SessionKey);
using AoeSetGeSessionFunc = AoeStatus (*)(SessionKey, ge::Session *);
using AoeSetDependGraphFunc = AoeStatus (*)(SessionKey, const std::vector<ge::Graph> &);
using AoeSetDependGraphsInputsFunc = AoeStatus (*)(SessionKey, const std::vector<std::vector<ge::Tensor>> &);
using AoeSetTuningGraphInputFunc = AoeStatus (*)(SessionKey, const std::vector<ge::Tensor> &);
using AoeSetTuningGraphFunc = AoeStatus (*)(SessionKey, const ge::Graph &);
using AoeTuningGraphFunc = AoeStatus (*)(SessionKey, const std::map<ge::AscendString, ge::AscendString> &);

struct AoeFunc {
  AoeInitializeFunc aoe_initialize = nullptr;
  AoeFinalizeFunc aoe_finalize = nullptr;
  AoeCreateSessionFunc aoe_create_session = nullptr;
  AoeDesstroySessionFunc aoe_destroy_session = nullptr;
  AoeSetGeSessionFunc aoe_set_gesession = nullptr;
  AoeSetDependGraphFunc aoe_set_dependgraphs = nullptr;
  AoeSetTuningGraphFunc aoe_set_tuninggraph = nullptr;
  AoeTuningGraphFunc aoe_tuning_graph = nullptr;
  AoeSetDependGraphsInputsFunc aoe_set_depend_graphs_inputs = nullptr;
  AoeSetTuningGraphInputFunc aoe_set_tuning_graph_input = nullptr;
};

class NpuAoe {
 public:
  NpuAoe() = default;
  ~NpuAoe();

  static NpuAoe &GetInstance();
  Status AoeTuningInitialize(const ge::AscendString &work_path, const ge::AscendString &job_type);
  Status RunAoeTuning(const ge::Graph &graph, const std::map<ge::AscendString, ge::AscendString> &options,
                      const std::vector<ge::Tensor> &example_inputs, void *stram, ge::Session *geSession);
  Status AoeTuningFinalize();

  NpuAoe(const NpuAoe&) = delete;
  NpuAoe(NpuAoe &&) = delete;
  NpuAoe& operator=(const NpuAoe&) = delete;
  NpuAoe& operator=(NpuAoe &&) = delete;

 private:
  Status LoadAoeFunc();

  AoeFunc aoe_func_;
  void *handle_ = nullptr;
  int64_t exec_num_ = 0;
};

} // namespace tng

#endif // TORCHAIR_TORCHAIR_INCLUDE_NPU_AOE_H