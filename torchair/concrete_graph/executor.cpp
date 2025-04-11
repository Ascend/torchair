#include "executor.h"

#include <utility>
#include "graph/tensor.h"
#include "graph/types.h"
#include "graph_data.h"

#include "torch/torch.h"
#include "session.h"
#include "checker.h"
#include "utils.h"
#include "logger.h"

namespace tng {

class CpuGraphExecutor : public Executor {
 public:
  explicit CpuGraphExecutor(std::shared_ptr<tng::GraphData> graph_data) : graph_data_(std::move(graph_data)){};
  Status AssembleInputs(const std::vector<const at::Tensor*> &inputs) override {
    if (is_first_run_) {
      inputs_holder_.resize(inputs.size());
      for (size_t i = 0U; i < inputs.size(); ++i) {
        TNG_RETURN_IF_ERROR(AtTensorToGeTensor(*inputs[i], inputs_holder_[i]));
        TNG_LOG(INFO) << "Assemble aten input " << i << " " << DebugString(*inputs[i]) << " to "
                      << DebugString(inputs_holder_[i]);
      }
    } else {
      TNG_ASSERT(inputs_holder_.size() == inputs.size());
      for (size_t i = 0U; i < inputs.size(); ++i) {
        TNG_RETURN_IF_ERROR(AssembleDataAndShapeToGe(*inputs[i], inputs_holder_[i]));
        TNG_LOG(INFO) << "Assemble aten input " << i << " " << DebugString(*inputs[i]) << " to "
                      << DebugString(inputs_holder_[i]);
      }
    }
    return Status::Success();
  }

  Status AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &outputs) {
    TNG_ASSERT(outputs.empty());
    return Status::Success();
  }

  Status Run(const std::vector<c10::optional<at::Tensor>> &torch_outputs,
             std::vector<at::Tensor> &outputs, void *stream) override {
    TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs));
    TNG_ASSERT(stream == nullptr);
    outputs_holder_.clear();
    TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, outputs_holder_, stream));
    outputs.resize(outputs_holder_.size());
    for (size_t i = 0; i < outputs_holder_.size(); ++i) {
      TNG_RETURN_IF_ERROR(GeTensorToAtTensor(outputs_holder_[i], outputs[i]));
      TNG_LOG(INFO) << "Assemble ge output " << i << " " << DebugString(outputs_holder_[i]) << " to "
                    << DebugString(outputs[i]);
    }
    outputs_holder_.clear();
    is_first_run_ = false;
    return Status::Success();
  }

 private:
  std::vector<ge::Tensor> inputs_holder_;
  std::vector<ge::Tensor> outputs_holder_;
  std::shared_ptr<GraphData> graph_data_;
};

Status Executor::Create(const std::shared_ptr<GraphData> &graph_data, std::unique_ptr<Executor> &executor) {
  TNG_ASSERT_NOTNULL(graph_data);
  if (graph_data->executor_type == ExecutorType::CPU) {
    executor = std::unique_ptr<Executor>(new CpuGraphExecutor(graph_data));
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    TNG_ASSERT(!creators_.empty(), "No executor creator registered");
    TNG_RETURN_IF_ERROR(creators_.rbegin()->second(graph_data, executor));
  }
  TNG_ASSERT_NOTNULL(executor);
  return Status::Success();
}

std::map<int, Executor::Creator> Executor::creators_;
std::mutex Executor::mutex_;

bool Executor::RegisterExecutorCreator(const Executor::Creator &creator, int32_t priority) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!creators_.insert(std::make_pair(priority, creator)).second) {
    TNG_LOG(ERROR) << "Executor creator with priority " << priority << " already exists";
    return false;
  }
  return true;
}
}  // namespace tng
