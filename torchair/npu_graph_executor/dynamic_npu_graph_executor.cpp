#include "dynamic_npu_graph_executor.h"
#include "external/graph/types.h"

#include <ATen/record_function.h>

#include <utility>
#include "AllocatorManager.h"
#include "acl/acl_rt.h"
#include "checker.h"
#include "logger.h"
#include "npu_utils.h"
#include "session.h"
#include "torch/torch.h"
#include "utils.h"

namespace tng {
namespace {

inline Status ResetTensorDataPtr(gert::Tensor &ge_tensor) {
  TNG_ASSERT_GE_OK(ge_tensor.MutableTensorData().Free());
  return Status::Success();
}

inline Status ResetTensorDataPtr(ge::Tensor &ge_tensor) {
  const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t *data) {};
  TNG_ASSERT_GE_OK(ge_tensor.SetData(nullptr, 0U, kDoNothing));
  return Status::Success();
}

template <typename T>
Status RefreshAtTensorFromGeTensor(std::vector<T> &outputs_holder,
                                   const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                                   std::vector<at::Tensor> &outputs) {
  RECORD_FUNCTION("RefreshAtTensorFromGeTensor", {});
  outputs.resize(outputs_holder.size());
  for (size_t i = 0U; i < outputs_holder.size(); ++i) {
    if (!assigned_outputs.empty() && assigned_outputs[i].has_value()) {
      outputs[i] = assigned_outputs[i].value();
      TNG_LOG(DEBUG) << "Refresh assigned torch output " << i << " " << DebugString(outputs[i])
                     << " from ge output " << DebugString(outputs_holder[i]);
    } else {
      TNG_RETURN_IF_ERROR(GeTensorToAtTensor(outputs_holder[i], outputs[i]));
      TNG_LOG(DEBUG) << "Refresh unfed torch output " << i << " " << DebugString(outputs[i])
                     << " from ge output " << DebugString(outputs_holder[i]);
    }
  }
  return Status::Success();
}

}  // namespace

Status DynamicNpuGraphExecutor::AssembleInputs(const std::vector<const at::Tensor*> &inputs) {
  RECORD_FUNCTION("AssembleInputs", std::vector<c10::IValue>({}));
  TNG_ASSERT(graph_data_->input_placements.size() == inputs.size());
  static bool enable_load_execute_graph =
      Session::GetInstance().IsFastLoadGraphSupported() && Session::GetInstance().IsFastExecuteGraphSupported();
  if (is_first_run_) {
    TNG_RETURN_IF_ERROR(AssembleFrozenOption(graph_data_->frozen_input_flag_list, inputs, graph_data_->load_options));
    if (enable_load_execute_graph) {
      return AssembleInputsInner(inputs, gert_inputs_holder_);
    } else {
      return AssembleInputsInner(inputs, inputs_holder_);
    }
  } else {
    if (enable_load_execute_graph) {
      return UpdateInputsInner(inputs, gert_inputs_holder_);
    } else {
      return UpdateInputsInner(inputs, inputs_holder_);
    }
  }
}

template <typename T>
Status DynamicNpuGraphExecutor::AssembleInputsInner(const std::vector<const at::Tensor*> &inputs,
                                                    std::vector<T> &input_holders) {
  input_holders.resize(inputs.size());
  host_input_holders_.resize(inputs.size());
  TNG_ASSERT(graph_data_->frozen_input_flag_list.size() == inputs.size());

  for (size_t i = 0U; i < inputs.size(); ++i) {
    TNG_ASSERT(CheckPlacement(graph_data_->input_placements[i], *inputs[i]),
               "Input %zu placement is incompatible with expected %d.", i,
               static_cast<int>(graph_data_->input_placements[i]));

    if (graph_data_->input_placements[i] == Placement::DEVICE) {
      TNG_RETURN_IF_ERROR(AtNpuTensorToGeTensor(*inputs[i], input_holders[i]));
    } else {
      if ((*inputs[i]).sizes().size() > 1U) {
        host_input_holders_[i] = at::empty((*inputs[i]).sizes(), (*inputs[i]).options().device(at::kPrivateUse1));
      }
      UpdateHostInput(*inputs[i], input_holders[i], host_input_holders_[i]);
    }
    TNG_LOG(DEBUG) << "Assemble aten input " << i << " " << DebugString(*inputs[i]) << " to "
                   << DebugString(input_holders[i]);
  }
  return Status::Success();
}

template <typename T>
Status DynamicNpuGraphExecutor::UpdateHostInput(const at::Tensor &input, T &input_holder,
                                                at::Tensor &host_input_holder, bool update_shape_flag) {
  const at::Tensor *tmp = &input;
  if (input.sizes().size() > 1U) {  // GE只支持1维或者Scalar的Host输入
    size_t dst_size = static_cast<size_t>(host_input_holder.numel() * host_input_holder.element_size());
    size_t src_size = static_cast<size_t>(input.numel() * input.element_size());
    if (src_size > 0U) {
      TNG_RETURN_IF_ERROR(
          H2DMemcpy(host_input_holder.data_ptr(), dst_size, input.data_ptr(), src_size, first_stream_));
    }
    tmp = &host_input_holder;
  }
  if (is_first_run_) {
    TNG_RETURN_IF_ERROR(AtTensorToGeTensor(*tmp, input_holder));
  } else if (update_shape_flag) {
    TNG_RETURN_IF_ERROR(AssembleDataAndShapeToGe(*tmp, input_holder));
  }
  return Status::Success();
}

template <typename T>
Status DynamicNpuGraphExecutor::UpdateInputsInner(const std::vector<const at::Tensor*> &inputs,
                                                  std::vector<T> &input_holders) {
  for (size_t i = 0U; i < inputs.size(); ++i) {
    if (graph_data_->frozen_input_flag_list[i]) {
      TNG_LOG(DEBUG) << "Frozen input " << i << " skip update";
      continue;
    }
    TNG_ASSERT(CheckPlacement(graph_data_->input_placements[i], *inputs[i]),
               "Input %zu placement is incompatible with expected %d.", i,
               static_cast<int>(graph_data_->input_placements[i]));
    if (graph_data_->input_placements[i] == Placement::DEVICE) {
      TNG_RETURN_IF_ERROR(AssembleDataAndStorageShapeToGe(*inputs[i], input_holders[i]));
    } else {
      bool update_shape_flag = true;
      if ((*inputs[i]).sizes().size() > 1U) {
        // if dynamo shape have change, need to update host_input_holders_
        if ((*inputs[i]).sizes() != host_input_holders_[i].sizes()) {
          host_input_holders_[i] = at::empty((*inputs[i]).sizes(), (*inputs[i]).options().device(at::kPrivateUse1));
        } else {
          update_shape_flag = false;
        }
      }
      TNG_LOG(DEBUG) << "Host input " << i << " need update shape " << update_shape_flag;
      UpdateHostInput(*inputs[i], input_holders[i], host_input_holders_[i], update_shape_flag);
    }
    TNG_LOG(DEBUG) << "Update aten input " << i << " " << DebugString(*inputs[i]) << " to "
                   << DebugString(input_holders[i]);
  }
  return Status::Success();
}

Status DynamicNpuGraphExecutor::AllocAndSetFixedMemory(void *stream, std::shared_ptr<GraphData> &graph_data) {
  TNG_LOG(DEBUG) << "Enter DynamicNpuGraphExecutor set fixed_mem_addr_";
  // Register allocator for GE before run, according to stream.
  auto allocator = AllocatorManager::GetInstance().EnsureAllocatorRegistered(stream);
  TNG_ASSERT_NOTNULL(allocator);
  TNG_ASSERT_NOTNULL(graph_data->summary);
  size_t fixed_mem_size = 0U;
  TNG_ASSERT_GE_OK(graph_data->summary->GetFixedFeatureMemorySize(fixed_mem_size));
  TNG_LOG(DEBUG) << "DynamicNpuGraphExecutor AllocAndSetFixedMemory get fixed_mem_size : " << fixed_mem_size;
  ge::MemBlock *block = std::dynamic_pointer_cast<NpuAllocator>(allocator)->MallocFeatureMemory(fixed_mem_size, true);
  TNG_ASSERT_NOTNULL(block);
  fixed_mem_addr_ = block;
  TNG_RETURN_IF_ERROR(Session::GetInstance().SetGraphFixedFeatureMemoryBase(graph_data->id, fixed_mem_addr_->GetAddr(),
                                                                            fixed_mem_addr_->GetSize()));
  return Status::Success();
}

template <typename T>
Status DynamicNpuGraphExecutor::AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                                                std::vector<T> &output_holders) {
  RECORD_FUNCTION("AssembleOutputs", {});
  if (is_first_run_) {
    output_holders.resize(graph_data_->output_dtypes.size());
    TNG_LOG(INFO) << "Graph output size is " << output_holders.size();
  }

  TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == output_holders.size());
  for (size_t i = 0U; i < output_holders.size(); ++i) {
    if (!assigned_outputs.empty() && assigned_outputs[i].has_value()) {
      if (is_first_run_) {
        TNG_RETURN_IF_ERROR(AtTensorToGeTensor(assigned_outputs[i].value(), output_holders[i]));
      } else {
        TNG_RETURN_IF_ERROR(AssembleDataAndShapeToGe(assigned_outputs[i].value(), output_holders[i]));
      }
      TNG_LOG(DEBUG) << "Assemble pre-assigned output " << i << " " << DebugString(assigned_outputs[i].value())
                     << " to " << DebugString(output_holders[i]);
    } else {
      // setting the i-th tensor data to nullptr and data_size 0 means
      // that the i-th tensor memory will be allocated and returned by GE.
      TNG_RETURN_IF_ERROR(ResetTensorDataPtr(output_holders[i]));
      TNG_LOG(DEBUG) << "Assemble unfed output " << i;
    }
  }

  return Status::Success();
}

Status DynamicNpuGraphExecutor::Run(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                                    std::vector<at::Tensor> &outputs, void *stream) {
  SetStageTime(ExecutorStage::kBegin);
  TNG_LOG(INFO) << "Dynamic npu graph executor start to run graph " << graph_data_->id;
  if (stream == nullptr) {
    TNG_RETURN_IF_ERROR(GetCurrentStream(&stream));
  }
  // Register allocator for GE before run, according to stream.
  AllocatorManager::GetInstance().EnsureAllocatorRegistered(stream);
  if (is_first_run_) {
    if (fixed_mem_addr_ == nullptr) {
      TNG_RETURN_IF_ERROR(AllocAndSetFixedMemory(stream, graph_data_));
    }
    first_stream_ = stream;
  } else {
    TNG_ASSERT(first_stream_ == stream, "Unsupport run graph with different stream.");
  }

  static bool enable_load_execute_graph =
      Session::GetInstance().IsFastLoadGraphSupported() && Session::GetInstance().IsFastExecuteGraphSupported();
  SetStageTime(ExecutorStage::kPre);

  if (enable_load_execute_graph) {
    TNG_RETURN_IF_ERROR(AssembleOutputs(assigned_outputs, gert_outputs_holder_));
    SetStageTime(ExecutorStage::kAssembleOutputs);
    if (is_first_run_) {
      TNG_RETURN_IF_ERROR(Session::GetInstance().FastLoadGraph(graph_data_->id, graph_data_->load_options, stream));
    }

    TNG_RETURN_IF_ERROR(
        Session::GetInstance().FastExecuteGraph(graph_data_->id, gert_inputs_holder_, gert_outputs_holder_, stream));
    SetStageTime(ExecutorStage::kRunGraph);
    TNG_RETURN_IF_ERROR(RefreshAtTensorFromGeTensor(gert_outputs_holder_, assigned_outputs, outputs));
  } else {
    TNG_RETURN_IF_ERROR(AssembleOutputs(assigned_outputs, outputs_holder_));
    SetStageTime(ExecutorStage::kAssembleOutputs);
    TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, outputs_holder_, stream));
    SetStageTime(ExecutorStage::kRunGraph);
    TNG_RETURN_IF_ERROR(RefreshAtTensorFromGeTensor(outputs_holder_, assigned_outputs, outputs));
  }

  TNG_LOG(EVENT) << "Dynamic executor call " << GenEventLog();
  TNG_LOG(INFO) << "Dynamic npu graph executor run graph " << graph_data_->id << " on stream " << stream
                 << " successfully.";
  is_first_run_ = false;
  return Status::Success();
}

DynamicNpuGraphExecutor::~DynamicNpuGraphExecutor() noexcept {
  auto stream_to_allocators = AllocatorManager::GetInstance().GetAllRegisteredAllocator();
  for (const auto &steam_allocator : stream_to_allocators) {
    if (fixed_mem_addr_ != nullptr) {
      std::dynamic_pointer_cast<NpuAllocator>(steam_allocator.second)->FreeFeatureMemory(fixed_mem_addr_, true);
    }
  }
  fixed_mem_addr_ = nullptr;
}

}  // namespace tng
