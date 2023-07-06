#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_GRAPH_DATA_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_GRAPH_DATA_H_

#include <memory>
#include "tng_status.h"

#include "external/graph/types.h"
#include "ge/ge_api.h"
#include "ge_ir.pb.h"
#include "graph/model_serialize.h"
#include "graph/utils/graph_utils_ex.h"

namespace tng {
enum class Placement {
  UNKNOWN = -1,
  HOST = 0,
  DEVICE = 1,
};
class GraphData {
 public:
  GraphData() {
    graph = nullptr;
    summary = nullptr;
  }
  ~GraphData() = default;
  uint32_t id = 0U;
  ge::proto::GraphDef graph_def;
  ge::GraphPtr graph = nullptr;
  std::map<ge::AscendString, ge::AscendString> compile_options;
  std::vector<Placement> input_placements;
  std::vector<ge::DataType> output_dtypes;
  std::shared_ptr<ge::CompiledGraphSummary> summary = nullptr;
};
}  // namespace tng
#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_GRAPH_DATA_H_