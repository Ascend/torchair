#ifndef TORCH_AIR_TORCH_AIR_MUTI_GEAR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_
#define TORCH_AIR_TORCH_AIR_MUTI_GEAR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_

#include "executor.h"
#include "graph_data.h"
#include "graph/tensor.h"
#include "memory/Allocator.h"
#include "static_npu_graph_executor.h"

#include <utility>

namespace tng {
class MutiGearNpuGraphExecutor : public StaticNpuGraphExecutor {
 public:
  explicit MutiGearNpuGraphExecutor(std::shared_ptr<tng::GraphData> graph_data)
      : StaticNpuGraphExecutor(std::move(graph_data)){};

  Status Run(const std::vector<at::Tensor> &torch_inputs, const std::vector<c10::optional<at::Tensor>> &torch_outputs,
             std::vector<at::Tensor> &outputs, void *stream) override;
 private:

  Status AssembleInputs(const std::vector<at::Tensor> &inputs, void *stream);

  Status AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                         std::vector<ge::MemBlock *> &output_gear_memblock, void *stream);
  std::vector<c10::ScalarType> output_torch_dtype_;
  std::vector<std::vector<int64_t>> input_gears_;
  std::vector<size_t> output_size_;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_MUTI_GEAR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_
