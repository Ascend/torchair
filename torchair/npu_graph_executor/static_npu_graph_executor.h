#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_

#include "executor.h"
#include "graph_data.h"
#include "graph/tensor.h"
#include "memory/Allocator.h"

namespace tng {
class StaticNpuGraphExecutor : public Executor {
 public:
  explicit StaticNpuGraphExecutor(std::shared_ptr<tng::GraphData> graph_data) : graph_data_(std::move(graph_data)){};

  Status Run(const std::vector<at::Tensor> &torch_inputs, const std::vector<c10::optional<at::Tensor>> &torch_outputs,
             std::vector<at::Tensor> &outputs, void *stream) override;

  ~StaticNpuGraphExecutor() override;

 private:
  Status AssembleInputs(const std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &retain_tmp_device_inputs);

  Status AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                         std::vector<at::Tensor> &outputs);

  Status AllocAndSetConstMemory(void *stream);

  Status AllocAndUpdateFeatureMemory(void *stream);

  std::vector<ge::Tensor> inputs_holder_;
  std::vector<ge::Tensor> outputs_holder_;
  std::shared_ptr<GraphData> graph_data_;
  std::unique_ptr<ge::MemBlock, DelMemBlockFunc> const_mem_addr_{nullptr};
  ge::MemBlock *feature_map_block_{nullptr};
  bool fm_refreshable_{false};
  bool is_first_run_{true};

  std::vector<ge::Shape> output_shapes_;
  std::vector<at::TensorOptions> output_options_;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_