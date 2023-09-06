#include "dynamic_npu_graph_executor.h"
#include "external/graph/types.h"

#include <utility>
#include "AllocatorManager.h"
#include "checker.h"
#include "graph/utils/type_utils.h"
#include "logger.h"
#include "npu_utils.h"
#include "session.h"
#include "torch/torch.h"
#include "utils.h"

namespace tng {

Status DynamicNpuGraphExecutor::AssembleInputs(const std::vector<at::Tensor> &inputs,
                                               std::vector<at::Tensor> &retain_tmp_device_inputs) {
  TNG_ASSERT(graph_data_->input_placements.size() == inputs.size());
  bool is_first_run = inputs_holder_.empty();
  if (is_first_run) {
    inputs_holder_.resize(inputs.size());
  }

  for (size_t i = 0U; i < inputs.size(); ++i) {
    if (graph_data_->input_placements[i] == Placement::DEVICE) {
      if (!inputs[i].device().is_privateuseone()) {
        return Status::Error("Input %zu placement %s is incompatible with expected PrivateUse1.", i,
                             DebugString(inputs[i].device()).c_str());
      }

      if (is_first_run) {
        TNG_RETURN_IF_ERROR(AtTensorToGeTensor(inputs[i], inputs_holder_[i]));
      } else {
        TNG_RETURN_IF_ERROR(AssembleDataAndShapeToGe(inputs[i], inputs_holder_[i]));
      }
      TNG_LOG(DEBUG) << "Assemble aten device input " << i << " " << DebugString(inputs[i]) << " to "
                     << DebugString(inputs_holder_[i]);
    } else if (graph_data_->input_placements[i] == Placement::HOST) {
      if (!inputs[i].device().is_cpu()) {
        return Status::Error("Input %zu placement %s is incompatible with expected CPU.", i,
                             DebugString(inputs[i].device()).c_str());
      }

      const at::Tensor *input = &inputs[i];
      if (inputs[i].sizes().size() > 1U) {  // GE只支持1维或者Scalar的Host输入
        TNG_LOG(DEBUG) << "Host input " << i << " " << DebugString(inputs[i]) << " need copy to device";
        // copy host input to device
        auto device_input = at::empty(inputs[i].sizes(), inputs[i].options().device(at::kPrivateUse1));
        device_input.copy_(inputs[i], true);
        // device_input is a temporary variable that will be destructed after leaving the scope,
        // and it is necessary to control its destruct timing.
        retain_tmp_device_inputs.emplace_back(std::move(device_input));
        input = &retain_tmp_device_inputs.back();
      }

      if (is_first_run) {
        TNG_RETURN_IF_ERROR(AtTensorToGeTensor(*input, inputs_holder_[i]));
      } else {
        TNG_RETURN_IF_ERROR(AssembleDataAndShapeToGe(*input, inputs_holder_[i]));
      }
      TNG_LOG(DEBUG) << "Assemble aten host input " << i << " " << DebugString(inputs[i]) << " to "
                     << DebugString(inputs_holder_[i]);
    } else {
      TNG_ASSERT(false, "Invalid Placement::UNKNOWN of input %zu.", i);
    }
  }

  return Status::Success();
}

Status DynamicNpuGraphExecutor::AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &outputs) {
  // TODO: fix this case if necessary
  // This interface is only needed when partial outputs are specified by user,
  // but it is not involved currently.
  TNG_ASSERT(outputs.empty());
  return Status::Success();
}

Status DynamicNpuGraphExecutor::Run(const std::vector<at::Tensor> &torch_inputs,
                                    const std::vector<c10::optional<at::Tensor>> &torch_outputs,
                                    std::vector<at::Tensor> &outputs, void *stream) {
  TNG_LOG(INFO) << "Dynamic npu graph executor start to run graph " << graph_data_->id;

  std::vector<at::Tensor> retain_tmp_device_inputs;
  TNG_RETURN_IF_ERROR(AssembleInputs(torch_inputs, retain_tmp_device_inputs));
  TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs));

  if (stream == nullptr) {
    TNG_RETURN_IF_ERROR(GetCurrentStream(&stream));
  }

  // Register allocator for GE before run, according to stream.
  AllocatorManager::GetInstance().EnsureAllocatorRegistered(stream);

  std::vector<ge::Tensor> ge_outputs;
  ge_outputs.resize(graph_data_->output_dtypes.size());
  TNG_LOG(INFO) << "Graph output size is " << ge_outputs.size();
  for (size_t i = 0; i < ge_outputs.size(); ++i) {
    const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t *data) {};
    // setting the i-th tensor data to nullptr means
    // that the i-th tensor memory will be allocated and returned by GE.
    TNG_ASSERT_GE_OK(ge_outputs[i].SetData(nullptr, 0U, kDoNothing));
  }

  TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, ge_outputs, stream));
  outputs.resize(ge_outputs.size());
  for (size_t i = 0; i < ge_outputs.size(); ++i) {
    TNG_RETURN_IF_ERROR(GeTensorToAtTensor(ge_outputs[i], outputs[i]));
    TNG_LOG(DEBUG) << "Assemble ge output " << i << " " << DebugString(ge_outputs[i]) << " to torch output "
                   << DebugString(outputs[i]);
  }

  retain_tmp_device_inputs.clear();
  ge_outputs.clear();
  TNG_LOG(DEBUG) << "Dynamic npu graph executor run graph " << graph_data_->id << " on stream " << stream
                 << " successfully.";
  return Status::Success();
}

}  // namespace tng