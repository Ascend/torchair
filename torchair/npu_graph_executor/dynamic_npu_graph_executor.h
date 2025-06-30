#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_DYNAMIC_NPU_GRAPH_EXECUTOR_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_DYNAMIC_NPU_GRAPH_EXECUTOR_H_

#include "exe_graph/runtime/tensor.h"
#include "executor.h"
#include "graph/tensor.h"
#include "graph_data.h"
#include "memory/Allocator.h"

namespace tng {
class DynamicNpuGraphExecutor : public Executor {
 public:
  explicit DynamicNpuGraphExecutor(std::shared_ptr<tng::GraphData> graph_data) : graph_data_(std::move(graph_data)){};

  Status Run(const std::vector<c10::optional<at::Tensor>> &assigned_outputs, std::vector<at::Tensor> &outputs,
             void *stream) override;

  Status AssembleInputs(const std::vector<const at::Tensor*> &inputs) override;

  ~DynamicNpuGraphExecutor() override;

 private:
  Status AllocAndSetFixedMemory(void *stream, std::shared_ptr<GraphData> &graph_data);

  template <typename T>
  Status AssembleInputsInner(const std::vector<const at::Tensor*> &inputs, std::vector<T> &input_holders);

  template <typename T>
  Status UpdateHostInput(const at::Tensor &input, T &input_holder,
                         at::Tensor &host_input_holder, bool update_shape_flag = false);

  template <typename T>
  Status UpdateInputsInner(const std::vector<const at::Tensor*> &inputs, std::vector<T> &input_holders);

  template <typename T>
  Status AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                         std::vector<T> &output_holders);

  ge::MemBlock *fixed_mem_addr_{nullptr};
  std::vector<ge::Tensor> inputs_holder_;
  std::vector<ge::Tensor> outputs_holder_;
  std::vector<gert::Tensor> gert_inputs_holder_;
  std::vector<gert::Tensor> gert_outputs_holder_;
  std::shared_ptr<GraphData> graph_data_;
  std::vector<at::Tensor> host_input_holders_;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_DYNAMIC_NPU_GRAPH_EXECUTOR_H_