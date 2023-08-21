#include "external/graph/types.h"
#include "static_npu_graph_executor.h"

#include <utility>
#include "checker.h"
#include "graph/utils/type_utils.h"
#include "logger.h"
#include "session.h"
#include "torch/torch.h"
#include "utils.h"
#include "npu_utils.h"

namespace tng {

Status StaticNpuGraphExecutor::AssembleInputs(const std::vector<at::Tensor> &inputs,
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
        // In static graph input shape remains unchanged, only data ptr need to be updated.
        TNG_RETURN_IF_ERROR(AssembleDataToGe(inputs[i], inputs_holder_[i]));
      }
      TNG_LOG(DEBUG) << "Assemble aten device input " << i << " " << DebugString(inputs[i]) << " to "
                     << DebugString(inputs_holder_[i]);
    } else if (graph_data_->input_placements[i] == Placement::HOST) {
      if (!inputs[i].device().is_cpu()) {
        return Status::Error("Input %zu placement %s is incompatible with expected CPU.", i,
                             DebugString(inputs[i].device()).c_str());
      }
      // copy host input to device
      auto device_input = at::empty(inputs[i].sizes(), inputs[i].options().device(at::kPrivateUse1));
      device_input.copy_(inputs[i], true);
      // device_input is a temporary variable that will be destructed after leaving the scope,
      // and it is necessary to control its destruct timing.
      retain_tmp_device_inputs.emplace_back(std::move(device_input));

      if (is_first_run) {
        TNG_RETURN_IF_ERROR(AtTensorToGeTensor(retain_tmp_device_inputs.back(), inputs_holder_[i]));
      } else {
        // In static graph input shape remains unchanged, only data ptr need to be updated.
        TNG_RETURN_IF_ERROR(AssembleDataToGe(retain_tmp_device_inputs.back(), inputs_holder_[i]));
      }
      TNG_LOG(DEBUG) << "Assemble aten host input " << i << " " << DebugString(retain_tmp_device_inputs.back())
                     << " to " << DebugString(inputs_holder_[i]);
    } else {
      TNG_ASSERT(false, "Invalid Placement::UNKNOWN of input %zu.", i);
    }
  }

  return Status::Success();
}

Status StaticNpuGraphExecutor::AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &outputs) {
  // TODO: fix this case if necessary
  // This interface is only needed when partial outputs are specified by user,
  // but it is not involved currently.
  TNG_ASSERT(outputs.empty());
  return Status::Success();
}

Status StaticNpuGraphExecutor::RefreshGraphOutputs(std::vector<at::Tensor> &outputs) {
  bool is_first_run = outputs_holder_.empty();
  if (is_first_run) {
    const std::vector<ge::DataType> &output_ge_dtypes = graph_data_->output_dtypes;
    TNG_ASSERT_GE_OK(graph_data_->summary->GetOutputShapes(output_shapes_));
    TNG_ASSERT(output_shapes_.size() == output_ge_dtypes.size());
    outputs_holder_.resize(output_shapes_.size());

    for (size_t i = 0U; i < output_ge_dtypes.size(); ++i) {
      c10::ScalarType output_i_torch_dtype;
      GeDtypeToAtDtype(output_ge_dtypes[i], output_i_torch_dtype);
      at::TensorOptions option = at::TensorOptions().dtype(output_i_torch_dtype).device(at::kPrivateUse1);
      output_options_.push_back(option);
    }
  }

  outputs.resize(output_shapes_.size());
  outputs.clear();
  for (size_t i = 0U; i < output_shapes_.size(); ++i) {
    auto output_i = at::empty(output_shapes_[i].GetDims(), output_options_[i]);
    outputs.push_back(output_i);

    if (is_first_run) {
      TNG_RETURN_IF_ERROR(AtTensorToGeTensor(output_i, outputs_holder_[i]));
    } else {
      TNG_RETURN_IF_ERROR(AssembleDataToGe(output_i, outputs_holder_[i]));
    }
    TNG_LOG(DEBUG) << "Assemble ge output " << i << " " << DebugString(output_i) << " to "
                   << DebugString(outputs_holder_[i]);
  }

  return Status::Success();
}

Status StaticNpuGraphExecutor::Run(const std::vector<at::Tensor> &torch_inputs,
                                   const std::vector<c10::optional<at::Tensor>> &torch_outputs,
                                   std::vector<at::Tensor> &outputs, void *stream) {
  std::vector<at::Tensor> retain_tmp_device_inputs;
  TNG_RETURN_IF_ERROR(AssembleInputs(torch_inputs, retain_tmp_device_inputs));

  TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs));

  TNG_RETURN_IF_ERROR(RefreshGraphOutputs(outputs));

  if (stream == nullptr) {
    TNG_RETURN_IF_ERROR(GetCurrentStream(&stream));
  }

  TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, outputs_holder_, stream));
  retain_tmp_device_inputs.clear();
  TNG_LOG(INFO) << "StaticNpuGraphExecutor::Run graph " << graph_data_->id << " on stream " << stream
                << " successfully.";
  return Status::Success();
}

}  // namespace tng