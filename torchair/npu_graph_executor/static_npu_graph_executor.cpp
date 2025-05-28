#include "static_npu_graph_executor.h"
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

Status StaticNpuGraphExecutor::AssembleInputs(const std::vector<const at::Tensor*> &inputs) {
  RECORD_FUNCTION("AssembleInputs", std::vector<c10::IValue>({}));
  TNG_ASSERT(graph_data_->input_placements.size() == inputs.size());
  static bool enable_load_execute_graph =
      Session::GetInstance().IsFastLoadGraphSupported() && Session::GetInstance().IsFastExecuteGraphSupported();
  if (is_first_run_) {
    TNG_RETURN_IF_ERROR(AssembleFrozenOption(graph_data_->frozen_input_flag_list, inputs, graph_data_->load_options));
    TNG_RETURN_IF_ERROR(AssembleHostInputOption(inputs, graph_data_->load_options));
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
Status StaticNpuGraphExecutor::AssembleInputsInner(const std::vector<const at::Tensor*> &inputs,
                                                   std::vector<T> &input_holders) {
  input_holders.resize(inputs.size());
  host_input_holders_.resize(inputs.size());
  TNG_ASSERT(graph_data_->frozen_input_flag_list.size() == inputs.size());

  for (size_t i = 0U; i < inputs.size(); ++i) {
    TNG_ASSERT(CheckPlacement(graph_data_->input_placements[i], *inputs[i]),
               "Input %zu placement is incompatible with expected %d.", i,
               static_cast<int>(graph_data_->input_placements[i]));
    if ((graph_data_->input_placements[i] == Placement::HOST) && !IsSupportHostInput()) {
      auto host_input_holder = at::empty((*inputs[i]).sizes(), (*inputs[i]).options().device(at::kPrivateUse1));
      size_t dst_size = static_cast<size_t>(host_input_holder.numel() * host_input_holder.element_size());
      size_t src_size = static_cast<size_t>((*inputs[i]).numel() * (*inputs[i]).element_size());
      auto copy_size = std::make_pair(src_size, dst_size);
      host_input_holders_[i] = std::make_pair(host_input_holder, copy_size);
      TNG_RETURN_IF_ERROR(AtTensorToGeTensor(host_input_holders_[i].first, input_holders[i]));
      if (host_input_holders_[i].second.first > 0) {
        TNG_RETURN_IF_ERROR(H2DMemcpy(host_input_holders_[i].first.data_ptr(), host_input_holders_[i].second.second,
                                      (*inputs[i]).data_ptr(), host_input_holders_[i].second.first, first_stream_));
      }
    } else {
      TNG_RETURN_IF_ERROR(AtNpuTensorToGeTensor(*inputs[i], input_holders[i]));
    }
    TNG_LOG(DEBUG) << "Assemble aten input " << i << " " << DebugString(*inputs[i]) << " to "
                   << DebugString(input_holders[i]);
  }
  return Status::Success();
}

template <typename T>
Status StaticNpuGraphExecutor::UpdateInputsInner(const std::vector<const at::Tensor*> &inputs,
                                                 std::vector<T> &input_holders) {
  for (size_t i = 0U; i < inputs.size(); ++i) {
    if (graph_data_->frozen_input_flag_list[i]) {
      TNG_LOG(DEBUG) << "Frozen input " << i << " skip update";
      continue;
    }
    TNG_ASSERT(CheckPlacement(graph_data_->input_placements[i], *inputs[i]),
               "Input %zu placement is incompatible with expected %d.", i,
               static_cast<int>(graph_data_->input_placements[i]));
    if ((graph_data_->input_placements[i] == Placement::HOST) && !IsSupportHostInput()) {
      if (host_input_holders_[i].second.first > 0) {
        TNG_RETURN_IF_ERROR(H2DMemcpy(host_input_holders_[i].first.data_ptr(), host_input_holders_[i].second.second,
                                      (*inputs[i]).data_ptr(), host_input_holders_[i].second.first, first_stream_));
      }
    } else {
      // In static graph input shape remains unchanged, only data ptr need to be updated.
      TNG_RETURN_IF_ERROR(AssembleDataToGe(*inputs[i], input_holders[i], false));
    }
    TNG_LOG(DEBUG) << "Update aten input " << i << " " << DebugString(*inputs[i]) << " to "
                   << DebugString(input_holders[i]);
  }
  return Status::Success();
}

template <typename T>
Status StaticNpuGraphExecutor::AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                                               std::vector<at::DataPtr> &data_ptrs,
                                               std::vector<T> &output_holders, void *stream) {
  RECORD_FUNCTION("AssembleOutputs", std::vector<c10::IValue>({}));
  if (is_first_run_) {
    auto output_ge_dtypes = graph_data_->output_dtypes;
    std::vector<ge::Shape> output_ge_shapes;
    TNG_ASSERT_GE_OK(graph_data_->summary->GetOutputShapes(output_ge_shapes));
    TNG_ASSERT_EQ(output_ge_shapes.size(), output_ge_dtypes.size());
    output_holders.resize(output_ge_dtypes.size());
    output_size_.resize(output_ge_dtypes.size());
    output_shapes_.resize(output_ge_dtypes.size());
    output_torch_dtype_.resize(output_ge_dtypes.size());

    TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == output_ge_dtypes.size());
    for (size_t i = 0U; i < output_ge_dtypes.size(); ++i) {
      TNG_RETURN_IF_ERROR(GeDtypeToAtDtype(output_ge_dtypes[i], output_torch_dtype_[i]));
      output_shapes_[i] = output_ge_shapes[i].GetDims();
      output_size_[i] =
        at::detail::computeStorageNbytesContiguous(output_shapes_[i], elementSize(output_torch_dtype_[i]));

      TNG_RETURN_IF_ERROR(UpdateTensorInfos(output_holders[i], output_shapes_[i], ge::FORMAT_ND, output_ge_dtypes[i]));
    }
  }

  TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == output_holders.size());
  auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
  TNG_ASSERT_NOTNULL(allocator);
  data_ptrs.resize(output_holders.size());
  for (size_t i = 0U; i < output_holders.size(); ++i) {
    if (!assigned_outputs.empty() && assigned_outputs[i].has_value()) {
      if (is_first_run_) {
        TNG_RETURN_IF_ERROR(AtNpuTensorToGeTensor(assigned_outputs[i].value(), output_holders[i]));
      } else {
        TNG_RETURN_IF_ERROR(AssembleDataAndStorageShapeToGe(assigned_outputs[i].value(), output_holders[i]));
      }
      TNG_LOG(DEBUG) << "Assemble pre-assigned output " << i << " " << DebugString(assigned_outputs[i].value())
                     << " to " << DebugString(output_holders[i]);
      continue;
    }
    auto data_ptr = allocator->allocate(output_size_[i]);
    if (output_size_[i] != 0) {
        TNG_ASSERT_NOTNULL(data_ptr);
    }
    data_ptrs[i] = std::move(data_ptr);
    TNG_RETURN_IF_ERROR(UpdateTensorData(output_holders[i], data_ptr.get(), output_size_[i]));
    TNG_LOG(DEBUG) << "Malloc " << output_size_[i] << " for ge output tensor.";
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
  if (!fm_refreshable_ && feature_map_block_ != nullptr) {
    TNG_LOG(INFO) << "No need to refresh feature map addr, use addr = " << feature_map_block_->GetAddr()
                  << " , size = " << feature_map_block_->GetSize();
    return Status::Success();
  }

  size_t fm_size = 0U;
  TNG_ASSERT_GE_OK(graph_data_->summary->GetFeatureMemorySize(fm_size));
  // Register allocator for GE before run, according to stream.
  auto allocator = AllocatorManager::GetInstance().EnsureAllocatorRegistered(stream);
  TNG_ASSERT_NOTNULL(allocator);
  TNG_LOG(INFO) << "Try to alloc and update feature map memory, graph id = " << graph_data_->id
                << " , size = " << fm_size;

  if (!fm_refreshable_) {
    // when refreshable fm is also needed to malloc for first run
    feature_map_block_ = std::dynamic_pointer_cast<NpuAllocator>(allocator)->MallocFeatureMemory(fm_size, false);
  } else {
    feature_map_block_ = std::dynamic_pointer_cast<NpuAllocator>(allocator)->Malloc(fm_size);
  }
  TNG_ASSERT_NOTNULL(feature_map_block_);
  TNG_ASSERT(
      Session::GetInstance()
          .UpdateGraphFeatureMemoryBase(graph_data_->id, feature_map_block_->GetAddr(), feature_map_block_->GetSize())
          .IsSuccess());

  TNG_LOG(INFO) << "AllocAndUpdateFeatureMemory success, feature map addr = " << feature_map_block_->GetAddr()
                << " , size = " << feature_map_block_->GetSize();
  return Status::Success();
}

Status StaticNpuGraphExecutor::AllocAndSetFixedMemory(void *stream, std::shared_ptr<GraphData> &graph_data) {
  TNG_LOG(DEBUG) << "Enter StaticNpuGraphExecutor and set  fixed_mem_addr_";
  // Register allocator for GE before run, according to stream.
  auto allocator = AllocatorManager::GetInstance().EnsureAllocatorRegistered(stream);
  TNG_ASSERT_NOTNULL(allocator);
  TNG_ASSERT_NOTNULL(graph_data->summary);
  size_t fixed_mem_size = 0U;
  TNG_ASSERT_GE_OK(graph_data->summary->GetFixedFeatureMemorySize(fixed_mem_size));
  TNG_LOG(DEBUG) << "StaticNpuGraphExecutor AllocAndSetFixedMemory get fixed_mem_size : " << fixed_mem_size;
  ge::MemBlock *block = std::dynamic_pointer_cast<NpuAllocator>(allocator)->MallocFeatureMemory(fixed_mem_size, true);
  TNG_ASSERT_NOTNULL(block);
  fixed_mem_addr_ = block;
  TNG_RETURN_IF_ERROR(Session::GetInstance().SetGraphFixedFeatureMemoryBase(graph_data->id, fixed_mem_addr_->GetAddr(),
                                                                            fixed_mem_addr_->GetSize()));
  return Status::Success();
}

Status StaticNpuGraphExecutor::Run(const std::vector<c10::optional<at::Tensor>> &torch_outputs,
                                   std::vector<at::Tensor> &outputs, void *stream) {
  SetStageTime(ExecutorStage::kBegin);
  std::vector<at::DataPtr> data_ptrs;
  if (stream == nullptr) {
    TNG_RETURN_IF_ERROR(GetCurrentStream(&stream));
  }

  if (is_first_run_) {
    TNG_ASSERT_GE_OK(graph_data_->summary->GetFeatureMemoryBaseRefreshable(fm_refreshable_));
    TNG_RETURN_IF_ERROR(AllocAndSetConstMemory(stream));
    first_stream_ = stream;
  } else {
    TNG_ASSERT(first_stream_ == stream, "Unsupport run graph with different stream.");
  }
  TNG_RETURN_IF_ERROR(AllocAndUpdateFeatureMemory(stream));

  static bool enable_load_execute_graph =
      Session::GetInstance().IsFastLoadGraphSupported() && Session::GetInstance().IsFastExecuteGraphSupported();
  SetStageTime(ExecutorStage::kPre);

  if (enable_load_execute_graph) {
    TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs, data_ptrs, gert_outputs_holder_, stream));
    SetStageTime(ExecutorStage::kAssembleOutputs);
    if (is_first_run_) {
      TNG_RETURN_IF_ERROR(Session::GetInstance().FastLoadGraph(graph_data_->id, graph_data_->load_options, stream));
    }

    TNG_RETURN_IF_ERROR(
        Session::GetInstance().FastExecuteGraph(graph_data_->id, gert_inputs_holder_, gert_outputs_holder_, stream));
    SetStageTime(ExecutorStage::kRunGraph);
  } else {
    TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs, data_ptrs, outputs_holder_, stream));
    SetStageTime(ExecutorStage::kAssembleOutputs);
    TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, outputs_holder_, stream));
    SetStageTime(ExecutorStage::kRunGraph);
  }

  outputs.clear();
  {
    RECORD_FUNCTION("RefreshAtTensorFromGeTensor", {});
    for (size_t i = 0U; i < data_ptrs.size(); i++) {
      if (!torch_outputs.empty() && torch_outputs[i].has_value()) {
        outputs.push_back(torch_outputs[i].value());
        continue;
      }
      outputs.push_back(
          MakeAtTensor(output_shapes_[i], output_torch_dtype_[i], output_size_[i], std::move(data_ptrs[i])));
      TNG_LOG(DEBUG) << "Refresh torch output " << i << " " << DebugString(outputs[i]);
    }
  }

  TNG_LOG(EVENT)  << "Static executor call " << GenEventLog();
  TNG_LOG(INFO) << "Static npu graph executor run graph " << graph_data_->id << " on stream " << stream
                << " successfully.";
  if (fm_refreshable_) {
    TNG_ASSERT_NOTNULL(feature_map_block_);
    feature_map_block_->Free();
    feature_map_block_ = nullptr;
  }
  is_first_run_ = false;
  return Status::Success();
}

StaticNpuGraphExecutor::~StaticNpuGraphExecutor() noexcept {
  auto stream_to_allocators = AllocatorManager::GetInstance().GetAllRegisteredAllocator();
  for (const auto &steam_allocator : stream_to_allocators) {
    if (feature_map_block_ != nullptr) {
      std::dynamic_pointer_cast<NpuAllocator>(steam_allocator.second)->FreeFeatureMemory(feature_map_block_, false);
    }
    if (fixed_mem_addr_ != nullptr) {
      std::dynamic_pointer_cast<NpuAllocator>(steam_allocator.second)->FreeFeatureMemory(fixed_mem_addr_, true);
    }
  }
  feature_map_block_ = nullptr;
  fixed_mem_addr_ = nullptr;
}

}  // namespace tng
