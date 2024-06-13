#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_CONCRETE_GRAPH_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_CONCRETE_GRAPH_H_

#include <map>

#include "executor.h"
#include "export.h"
#include "external/graph/ascend_string.h"
#include "external/graph/types.h"
#include "graph/tensor.h"
#include "tng_status.h"
#include "torch/torch.h"

namespace tng {
class GraphData;
class NpuConcreteGraph {
 public:
  static Status Create(const void *serialized_proto, size_t proto_size,
                       const std::map<ge::AscendString, ge::AscendString> &options,
                       std::vector<int64_t> input_placements, std::vector<int64_t> output_dtypes, int64_t executor_type,
                       std::unique_ptr<NpuConcreteGraph> &graph);
  Status Compile();
  Status AutoTune(const std::vector<at::Tensor> &example_inputs, void *stream = nullptr);
  Status SetHintShape(const std::vector<std::vector<int64_t>> &inputs_shape,
                      const std::vector<std::vector<int64_t>> &outputs_shape);
  Status Run(const std::vector<at::Tensor> &torch_inputs, const std::vector<c10::optional<at::Tensor>> &torch_outputs,
             std::vector<at::Tensor> &outputs, void *stream = nullptr);

  static Status ReleaseResource();
  static Status InitializeResource(const std::map<std::string, std::string> &options);

 private:
  NpuConcreteGraph(std::shared_ptr<GraphData> graph_data);
  std::shared_ptr<GraphData> graph_data_ = nullptr;
  std::unique_ptr<Executor> executor_ = nullptr;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_CONCRETE_GRAPH_H_