#include "external/graph/types.h"
#include "static_npu_graph_executor.h"

#include <ATen/record_function.h>

#include <utility>
#include "checker.h"
#include "graph/utils/type_utils.h"
#include "logger.h"
#include "session.h"
#include "torch/torch.h"
#include "utils.h"
#include "npu_utils.h"
#include "AllocatorManager.h"

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
      TNG_LOG(DEBUG) << "Assemble aten host input " << i << " " << DebugString(inputs[i]) << " to "
                     << DebugString(inputs_holder_[i]);
    } else {
      TNG_ASSERT(false, "Invalid Placement::UNKNOWN of input %zu.", i);
    }
  }

  return Status::Success();
}

Status StaticNpuGraphExecutor::AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                                               std::vector<at::Tensor> &outputs) {
  bool is_first_run = outputs_holder_.empty();
  if (is_first_run) {
    const std::vector<ge::DataType> &output_ge_dtypes = graph_data_->output_dtypes;
    TNG_ASSERT_GE_OK(graph_data_->summary->GetOutputShapes(output_shapes_));
    TNG_ASSERT_EQ(output_shapes_.size(), output_ge_dtypes.size());
    outputs_holder_.resize(output_shapes_.size());
    output_options_.resize(output_shapes_.size());

    TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == output_shapes_.size());
    for (size_t i = 0U; i < output_ge_dtypes.size(); ++i) {
      if (assigned_outputs.empty() || !assigned_outputs[i].has_value()) {
        c10::ScalarType output_i_torch_dtype = c10::ScalarType::Float;
        TNG_RETURN_IF_ERROR(GeDtypeToAtDtype(output_ge_dtypes[i], output_i_torch_dtype));
        at::TensorOptions option = at::TensorOptions().dtype(output_i_torch_dtype).device(at::kPrivateUse1);
        output_options_[i] = option;
      }
    }
  }

  TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == outputs_holder_.size());
  outputs.clear();
  for (size_t i = 0U; i < output_shapes_.size(); ++i) {
    if (!assigned_outputs.empty() && assigned_outputs[i].has_value()) {
      outputs.push_back(assigned_outputs[i].value());
      TNG_LOG(DEBUG) << "Assemble pre-assigned output " << i << " " << DebugString(outputs.back());
    } else {
      outputs.push_back(at::empty(output_shapes_[i].GetDims(), output_options_[i]));
      TNG_LOG(DEBUG) << "Create empty output " << i << " " << DebugString(outputs.back());
    }
    auto &output_i = outputs.back();
    if (is_first_run) {
      TNG_RETURN_IF_ERROR(AtTensorToGeTensor(output_i, outputs_holder_[i]));
    } else {
      TNG_RETURN_IF_ERROR(AssembleDataToGe(output_i, outputs_holder_[i]));
    }
    TNG_LOG(DEBUG) << "Assemble torch output " << i << " " << DebugString(output_i) << " to "
                   << DebugString(outputs_holder_[i]);
  }

  return Status::Success();
}

Status StaticNpuGraphExecutor::AllocAndSetConstMemory(void *stream) {
  // Register allocator for GE before run, according to stream.
  auto allocator = AllocatorManager::GetInstance().EnsureAllocatorRegistered(stream);
  TNG_ASSERT_NOTNULL(allocator);
  TNG_ASSERT_NOTNULL(graph_data_->summary);
  size_t const_mem_size = 0U;
  TNG_ASSERT_GE_OK(graph_data_->summary->GetConstMemorySize(const_mem_size));
  ge::MemBlock *block = allocator->Malloc(const_mem_size);
  TNG_ASSERT_NOTNULL(block);
  const_mem_addr_.reset(block);
  static auto kFree = [](ge::MemBlock *p) {
    if (p != nullptr) {
      p->Free();
      p = nullptr;
    }
  };
  const_mem_addr_.get_deleter() = kFree;
  TNG_RETURN_IF_ERROR(Session::GetInstance().SetGraphConstMemoryBase(graph_data_->id, const_mem_addr_->GetAddr(),
                                                                     const_mem_addr_->GetSize()));
  return Status::Success();
}

Status StaticNpuGraphExecutor::AllocAndUpdateFeatureMemory(void *stream) {
  size_t fm_size = 0U;
  TNG_ASSERT_GE_OK(graph_data_->summary->GetFeatureMemorySize(fm_size));
  // Register allocator for GE before run, according to stream.
  auto allocator = AllocatorManager::GetInstance().EnsureAllocatorRegistered(stream);
  TNG_ASSERT_NOTNULL(allocator);
  TNG_LOG(INFO) << "Try to alloc and update feature map memory, graph id = " << graph_data_->id
                << " , size = " << fm_size;
  bool is_fm_refreshable = false;
  TNG_ASSERT_GE_OK(graph_data_->summary->GetFeatureMemoryBaseRefreshable(is_fm_refreshable));
  if (!is_fm_refreshable) {
    if (feature_map_block_ == nullptr) {  // first run with fm is unable to refresh
      feature_map_block_ = std::dynamic_pointer_cast<NpuAllocator>(allocator)->MallocFeatureMemory(fm_size, nullptr);
      TNG_ASSERT_NOTNULL(feature_map_block_);
      TNG_ASSERT(Session::GetInstance()
                     .UpdateGraphFeatureMemoryBase(graph_data_->id, feature_map_block_->GetAddr(),
                                                   feature_map_block_->GetSize()).IsSuccess());
    } else {
      TNG_LOG(INFO) << "No need to refresh feature map addr, use addr = " << feature_map_block_->GetAddr()
                    << " , size = " << feature_map_block_->GetSize();
    }
  } else {
    feature_map_block_ = std::dynamic_pointer_cast<NpuAllocator>(allocator)->Malloc(fm_size);
    TNG_ASSERT_NOTNULL(feature_map_block_);
    TNG_ASSERT(Session::GetInstance()
            .UpdateGraphFeatureMemoryBase(graph_data_->id, feature_map_block_->GetAddr(), feature_map_block_->GetSize())
            .IsSuccess());
  }
  TNG_LOG(INFO) << "AllocAndUpdateFeatureMemory success, feature map addr = " << feature_map_block_->GetAddr()
                << " , size = " << feature_map_block_->GetSize();
  return Status::Success();
}

Status StaticNpuGraphExecutor::Run(const std::vector<at::Tensor> &torch_inputs,
                                   const std::vector<c10::optional<at::Tensor>> &torch_outputs,
                                   std::vector<at::Tensor> &outputs, void *stream) {
  std::vector<at::Tensor> retain_tmp_device_inputs;
  {
    RECORD_FUNCTION("AssembleInputs", std::vector<c10::IValue>({}));
    TNG_RETURN_IF_ERROR(AssembleInputs(torch_inputs, retain_tmp_device_inputs));
  }
  {
    RECORD_FUNCTION("AssembleOutputs", std::vector<c10::IValue>({}));
    TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs, outputs));
  }

  if (stream == nullptr) {
    TNG_RETURN_IF_ERROR(GetCurrentStream(&stream));
  }

  // Alloc const memory first run
  if (const_mem_addr_ == nullptr) {
    TNG_RETURN_IF_ERROR(AllocAndSetConstMemory(stream));
  }
  // Alloc and update feature memory
  TNG_RETURN_IF_ERROR(AllocAndUpdateFeatureMemory(stream));
  {
    RECORD_FUNCTION("RunGraphWithStreamAsync", std::vector<c10::IValue>({}));
    TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, outputs_holder_, stream));
  }
  retain_tmp_device_inputs.clear();
  TNG_LOG(INFO) << "StaticNpuGraphExecutor::Run graph " << graph_data_->id << " on stream " << stream
                << " successfully.";
  bool is_fm_refreshable = false;
  TNG_ASSERT_GE_OK(graph_data_->summary->GetFeatureMemoryBaseRefreshable(is_fm_refreshable));
  if (is_fm_refreshable) {
    TNG_ASSERT_NOTNULL(feature_map_block_);
    feature_map_block_->Free();
    feature_map_block_ = nullptr;
  }
  return Status::Success();
}

StaticNpuGraphExecutor::~StaticNpuGraphExecutor() noexcept {
  auto stream_to_allocators = AllocatorManager::GetInstance().GetAllRegisteredAllocator();
  for (const auto &steam_allocator : stream_to_allocators) {
    if (feature_map_block_ != nullptr) {
      std::dynamic_pointer_cast<NpuAllocator>(steam_allocator.second)->FreeFeatureMemory(feature_map_block_);
    }
  }
  feature_map_block_ = nullptr;
}

}  // namespace tng