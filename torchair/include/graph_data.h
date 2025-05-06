#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_GRAPH_DATA_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_GRAPH_DATA_H_

#include <memory>
#include "tng_status.h"

#include "external/graph/types.h"
#include "ge/ge_api.h"

namespace tng {
static const int64_t UNKONWN_DIM = -1;
static const int64_t UNKONWN_DIM_NUM = -2;

enum class Placement {
  UNKNOWN = -1,
  HOST = 0,
  DEVICE = 1,
};

enum class ExecutorType {
  UNKNOWN = -1,
  CPU = 0,
  NPU = 1,
};

class GraphData {
 public:
  GraphData() {
    graph = nullptr;
    summary = nullptr;
  }
  ~GraphData() = default;
  uint32_t id = 0U;
  ge::GraphPtr graph = nullptr;
  std::map<ge::AscendString, ge::AscendString> compile_options;
  std::map<ge::AscendString, ge::AscendString> load_options;
  std::vector<Placement> input_placements;
  std::vector<std::vector<int64_t>> inputs_shape;
  std::vector<std::vector<int64_t>> outputs_shape;
  std::vector<ge::DataType> output_dtypes;
  ExecutorType executor_type = ExecutorType::UNKNOWN;
  std::shared_ptr<ge::CompiledGraphSummary> summary = nullptr;
  int32_t deterministic_value = 1;
  std::vector<bool> frozen_input_flag_list;
};
}  // namespace tng
#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_GRAPH_DATA_H_