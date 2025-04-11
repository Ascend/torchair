#ifndef TORCH_AIR_TORCH_AIR_MUTI_GEAR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_
#define TORCH_AIR_TORCH_AIR_MUTI_GEAR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_

#include "executor.h"
#include "graph/tensor.h"
#include "graph_data.h"
#include "memory/Allocator.h"
#include "static_npu_graph_executor.h"

#include <utility>

namespace tng {
class MutiGearNpuGraphExecutor : public StaticNpuGraphExecutor {
 public:
  explicit MutiGearNpuGraphExecutor(std::shared_ptr<tng::GraphData> graph_data)
      : StaticNpuGraphExecutor(std::move(graph_data)){};

  Status Run(const std::vector<c10::optional<at::Tensor>> &torch_outputs,
             std::vector<at::Tensor> &outputs, void *stream) override;

  Status AssembleInputs(const std::vector<const at::Tensor*> &inputs) override;

 private:
  template <typename T>
  Status AssembleInputsInner(const std::vector<const at::Tensor*> &inputs, std::vector<T> &input_holders);

  template <typename T>
  Status UpdateInputsInner(const std::vector<const at::Tensor*> &inputs, std::vector<T> &input_holders);

  std::vector<std::vector<int64_t>> input_gears_;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_MUTI_GEAR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_
