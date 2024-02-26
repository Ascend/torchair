#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_DYNAMIC_NPU_GRAPH_EXECUTOR_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_DYNAMIC_NPU_GRAPH_EXECUTOR_H_

#include "executor.h"
#include "graph_data.h"
#include "graph/tensor.h"
#include "memory/Allocator.h"

namespace tng {
class DynamicNpuGraphExecutor : public Executor {
 public:
  explicit DynamicNpuGraphExecutor(std::shared_ptr<tng::GraphData> graph_data) : graph_data_(std::move(graph_data)){};

  Status Run(const std::vector<at::Tensor> &torch_inputs,
             const std::vector<c10::optional<at::Tensor>> &assigned_outputs, std::vector<at::Tensor> &outputs,
             void *stream) override;
  ~DynamicNpuGraphExecutor() override;

 private:
  Status AllocAndSetFixedMemory(void *stream, std::shared_ptr<GraphData> &graph_data);
  Status AssembleInputs(const std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &retain_tmp_device_inputs);
  ge::MemBlock *fixed_mem_addr_{nullptr};
  std::vector<ge::Tensor> inputs_holder_;
  std::shared_ptr<GraphData> graph_data_;
  bool is_first_run_{true};
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_DYNAMIC_NPU_GRAPH_EXECUTOR_H_