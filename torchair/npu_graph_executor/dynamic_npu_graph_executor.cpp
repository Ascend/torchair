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

template <typename T>
Status DynamicNpuGraphExecutor::AssembleInputs(const std::vector<at::Tensor> &inputs, std::vector<T> &input_holders,
                                               void *stream) {
  RECORD_FUNCTION("AssembleInputs", {});
  TNG_ASSERT(graph_data_->input_placements.size() == inputs.size());
  bool is_first_run = input_holders.empty();
  if (is_first_run) {
    input_holders.resize(inputs.size());
    host_input_holders_.resize(inputs.size());
    TNG_ASSERT(graph_data_->frozen_input_flag_list.size() == inputs.size());
    first_stream = stream;
  }

  for (size_t i = 0U; i < inputs.size(); ++i) {
    if (!is_first_run && graph_data_->frozen_input_flag_list[i]) {
      TNG_LOG(DEBUG) << "frozen input " << i << " " << DebugString(inputs[i])
                     << ", does not need to assemble if not in first run.";
      continue;
    }
    if (graph_data_->input_placements[i] == Placement::DEVICE) {
      if (!inputs[i].device().is_privateuseone()) {
        return Status::Error("Input %zu placement %s is incompatible with expected PrivateUse1.", i,
                             DebugString(inputs[i].device()).c_str());
      }

      if (is_first_run) {
        TNG_RETURN_IF_ERROR(AtNpuTensorToGeTensor(inputs[i], input_holders[i]));
      } else {
        TNG_RETURN_IF_ERROR(AssembleDataAndStorageShapeToGe(inputs[i], input_holders[i]));
      }
      TNG_LOG(DEBUG) << "Assemble aten device input " << i << " " << DebugString(inputs[i]) << " to "
                     << DebugString(input_holders[i]);
    } else if (graph_data_->input_placements[i] == Placement::HOST) {
      if (!inputs[i].device().is_cpu()) {
        return Status::Error("Input %zu placement %s is incompatible with expected CPU.", i,
                             DebugString(inputs[i].device()).c_str());
      }
      const at::Tensor *input = &inputs[i];
      bool update_shape_flag = true;
      if (inputs[i].sizes().size() > 1U) {  // GE只支持1维或者Scalar的Host输入
        TNG_LOG(DEBUG) << "Host input " << i << " " << DebugString(inputs[i]) << " need copy to device";
        // if dynamo shape have change,need to update host_input_holders_
        if (inputs[i].sizes() != host_input_holders_[i].sizes()) {
          host_input_holders_[i] = at::empty(inputs[i].sizes(), inputs[i].options().device(at::kPrivateUse1));
        } else {
          update_shape_flag = false;
        }
        size_t dst_size = static_cast<size_t>(host_input_holders_[i].numel() * host_input_holders_[i].element_size());
        size_t src_size = static_cast<size_t>(inputs[i].numel() * inputs[i].element_size());
        if (src_size > 0) {
          TNG_ASSERT(first_stream == stream,
                     "When the Tensor input is located on the host, the backend cannot support host input. "
                     "It is necessary to perform an H2D copy of the data before proceeding with asynchronous dispatch. "
                     "During the H2D copy of the input data, it is a synchronous copy without a stream, "
                     "while dispatching is an asynchronous operation with a stream. "
                     "To prevent the data copied to the device from being erroneously refreshed due to stream switching"
                     ", when there is host input, switching to different streams is not supported "
                     "during each execution.");
          auto stream_ret = aclrtSynchronizeStream(stream);
          TNG_ASSERT(stream_ret == ACL_ERROR_NONE, "ACL sync stream failed, return %d", stream_ret);
          auto ret = aclrtMemcpy(host_input_holders_[i].data_ptr(), dst_size, inputs[i].data_ptr(), src_size,
                                 ACL_MEMCPY_HOST_TO_DEVICE);
          TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL memory copy failed, return %d", ret);
        }
        input = &host_input_holders_[i];
      }
      if (is_first_run) {
        TNG_RETURN_IF_ERROR(AtTensorToGeTensor(*input, input_holders[i]));
      } else if (update_shape_flag) {
        TNG_RETURN_IF_ERROR(AssembleDataAndShapeToGe(*input, input_holders[i]));
      }
      TNG_LOG(DEBUG) << "Assemble aten host input " << i << " " << DebugString(inputs[i]) << " to "
                     << DebugString(input_holders[i]);
    } else {
      TNG_ASSERT(false, "Invalid Placement::UNKNOWN of input %zu.", i);
    }
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
  bool is_first_run = output_holders.empty();
  if (is_first_run) {
    output_holders.resize(graph_data_->output_dtypes.size());
    TNG_LOG(INFO) << "Graph output size is " << output_holders.size();
  }

  TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == output_holders.size());
  for (size_t i = 0U; i < output_holders.size(); ++i) {
    if (!assigned_outputs.empty() && assigned_outputs[i].has_value()) {
      if (is_first_run) {
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

Status DynamicNpuGraphExecutor::Run(const std::vector<at::Tensor> &torch_inputs,
                                    const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                                    std::vector<at::Tensor> &outputs, void *stream) {
  SetStageTime(ExecutorStage::kBegin);
  TNG_LOG(INFO) << "Dynamic npu graph executor start to run graph " << graph_data_->id;
  if (stream == nullptr) {
    TNG_RETURN_IF_ERROR(GetCurrentStream(&stream));
  }
  // Register allocator for GE before run, according to stream.
  AllocatorManager::GetInstance().EnsureAllocatorRegistered(stream);
  if (is_first_run_ && fixed_mem_addr_ == nullptr) {
    TNG_RETURN_IF_ERROR(AllocAndSetFixedMemory(stream, graph_data_));
  }

  static bool enable_load_execute_graph =
      Session::GetInstance().IsFastLoadGraphSupported() && Session::GetInstance().IsFastExecuteGraphSupported();
  SetStageTime(ExecutorStage::kPre);

  if (enable_load_execute_graph) {
    TNG_RETURN_IF_ERROR(AssembleInputs(torch_inputs, gert_inputs_holder_, stream));
    SetStageTime(ExecutorStage::kAssembleInputs);
    TNG_RETURN_IF_ERROR(AssembleOutputs(assigned_outputs, gert_outputs_holder_));
    SetStageTime(ExecutorStage::kAssembleOutputs);
    if (is_first_run_) {
      std::map<ge::AscendString, ge::AscendString> load_options;
      std::string frozen_option_value;
      TNG_RETURN_IF_ERROR(AssembleFrozenOption(graph_data_->frozen_input_flag_list, torch_inputs, frozen_option_value));
      if (frozen_option_value != "") {
        load_options[ge::AscendString("ge.exec.frozenInputIndexes")] = ge::AscendString(frozen_option_value.c_str());
      }
      TNG_RETURN_IF_ERROR(Session::GetInstance().FastLoadGraph(graph_data_->id, load_options, stream));
    }

    TNG_RETURN_IF_ERROR(
        Session::GetInstance().FastExecuteGraph(graph_data_->id, gert_inputs_holder_, gert_outputs_holder_, stream));
    SetStageTime(ExecutorStage::kRunGraph);
    TNG_RETURN_IF_ERROR(RefreshAtTensorFromGeTensor(gert_outputs_holder_, assigned_outputs, outputs));
  } else {
    TNG_RETURN_IF_ERROR(AssembleInputs(torch_inputs, inputs_holder_, stream));

    TNG_RETURN_IF_ERROR(AssembleOutputs(assigned_outputs, outputs_holder_));

    TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, outputs_holder_, stream));

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
