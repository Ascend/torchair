#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tensor.h"
#include "executor.h"
#include "graph/tensor.h"
#include "graph_data.h"
#include "memory/Allocator.h"

namespace tng {
class StaticNpuGraphExecutor : public Executor {
 public:
  explicit StaticNpuGraphExecutor(std::shared_ptr<tng::GraphData> graph_data) : graph_data_(std::move(graph_data)){};

  Status Run(const std::vector<c10::optional<at::Tensor>> &torch_outputs,
             std::vector<at::Tensor> &outputs, void *stream) override;

  Status AssembleInputs(const std::vector<const at::Tensor*> &inputs) override;

  ~StaticNpuGraphExecutor() override;

 private:
  Status AllocAndSetFixedMemory(void *stream, std::shared_ptr<GraphData> &graph_data);

  template <typename T>
  Status AssembleInputsInner(const std::vector<const at::Tensor*> &inputs, std::vector<T> &input_holders);

  template <typename T>
  Status UpdateInputsInner(const std::vector<const at::Tensor*> &inputs, std::vector<T> &input_holders);

 protected:
  Status AllocAndSetConstMemory(void *stream);

  Status AllocAndUpdateFeatureMemory(void *stream);

  template <typename T>
  Status AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                         std::vector<at::DataPtr> &data_ptrs, std::vector<T> &output_holders, void *stream);

  std::vector<ge::Tensor> inputs_holder_;
  std::vector<ge::Tensor> outputs_holder_;
  std::vector<gert::Tensor> gert_inputs_holder_;
  std::vector<gert::Tensor> gert_outputs_holder_;
  std::shared_ptr<GraphData> graph_data_;
  std::unique_ptr<ge::MemBlock, DelMemBlockFunc> const_mem_addr_{nullptr};
  ge::MemBlock *fixed_mem_addr_{nullptr};
  ge::MemBlock *feature_map_block_{nullptr};
  bool fm_refreshable_{false};

  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<std::pair<at::Tensor, std::pair<size_t, size_t>>> host_input_holders_;

  std::vector<c10::ScalarType> output_torch_dtype_;
  std::vector<size_t> output_size_;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_GRAPH_EXECUTOR_H_