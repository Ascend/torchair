#include "muti_gear_npu_graph_executor.h"
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
#include "torch_npu/inc/core/NPUFormat.h"
#include "utils.h"

namespace tng {
namespace {
Status ParseInputGears(const std::vector<const at::Tensor*> &inputs, std::vector<std::vector<int64_t>> &input_gears,
                       const std::vector<std::vector<int64_t>> &inputs_shape) {
  if (inputs_shape.empty()) {
    for (const auto input : inputs) {
      input_gears.emplace_back((*input).dim(), -1);
    }
    TNG_LOG(DEBUG) << "Parse input_gears but inputs_shape is empty, so construct it as " << DebugString(input_gears);
    return Status::Success();
  }
  for (size_t i = 0U; i < inputs.size(); ++i) {
    if (inputs_shape[i].size() == 1 && inputs_shape[i][0] == UNKONWN_DIM_NUM) {
      input_gears.emplace_back((*inputs[i]).dim(), -1);
      continue;
    }
    std::vector<int64_t> gear_dims;
    for (size_t dim_index = 0U; dim_index < inputs_shape[i].size(); ++dim_index) {
      if (inputs_shape[i][dim_index] == -1) {
        gear_dims.push_back(static_cast<int64_t>(dim_index));
      }
    }
    input_gears.emplace_back(gear_dims);
  }

  TNG_LOG(DEBUG) << "Parse input_gears " << DebugString(input_gears) << " from input shape "
                 << DebugString(inputs_shape);
  return Status::Success();
}

// 只刷新特定维度的shape to ge tensor
inline Status UpdateSpecificDims(ge::Tensor &ge_tensor, const at::Tensor &at_tensor,
                                 const std::vector<int64_t> &input_gear) {
  auto torch_size = at_tensor.sizes();
  for (auto dim : input_gear) {
    TNG_ASSERT_GE_OK(ge_tensor.SetShapeDim(dim, torch_size[dim]));
  }
  return Status::Success();
}

inline Status UpdateSpecificDims(gert::Tensor &ge_tensor, const at::Tensor &at_tensor,
                                 const std::vector<int64_t> &input_gear) {
  auto torch_size = at_tensor.sizes();
  for (auto dim : input_gear) {
    ge_tensor.MutableOriginShape().SetDim(dim, torch_size[dim]);
    ge_tensor.MutableStorageShape().SetDim(dim, torch_size[dim]);
  }
  return Status::Success();
}

template <typename T>
inline Status RefreshOutputShape(std::vector<std::vector<int64_t>> &real_output_shape,
                                 std::vector<T> &outputs_holder) {
  TNG_ASSERT_EQ(real_output_shape.size(), outputs_holder.size());
  for (size_t i = 0U; i < real_output_shape.size(); ++i) {
    TNG_RETURN_IF_ERROR(GetShapeFromGeTensor(real_output_shape[i], outputs_holder[i]));
  }
  return Status::Success();
}
}  // namespace

Status MutiGearNpuGraphExecutor::AssembleInputs(const std::vector<const at::Tensor*> &inputs) {
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
Status MutiGearNpuGraphExecutor::AssembleInputsInner(const std::vector<const at::Tensor*> &inputs,
                                                     std::vector<T> &input_holders) {
  input_holders.resize(inputs.size());
  host_input_holders_.resize(inputs.size());
  TNG_RETURN_IF_ERROR(ParseInputGears(inputs, input_gears_, graph_data_->inputs_shape));
  TNG_ASSERT(graph_data_->frozen_input_flag_list.size() == inputs.size());
  for (size_t i = 0U; i < inputs.size(); ++i) {
    TNG_ASSERT(CheckPlacement(graph_data_->input_placements[i], *inputs[i]),
               "Input %zu placement is incompatible with expected %d.", i,
               static_cast<int>(graph_data_->input_placements[i]));

    if (graph_data_->input_placements[i] == Placement::DEVICE) {
      TNG_ASSERT((input_gears_[i].empty() or IsBaseFormat(ge::Format(at_npu::native::get_npu_format(*inputs[i])))),
                 "Gear input expect format is base format not private format, but got format is %s.",
                 ge::GetFormatName(ge::Format(at_npu::native::get_npu_format(*inputs[i]))));
      TNG_RETURN_IF_ERROR(AtNpuTensorToGeTensor(*inputs[i], input_holders[i]));
    } else {
      TNG_ASSERT(input_gears_[i].empty(), "CPU tensor unsupport set gears");
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
    }
    TNG_LOG(DEBUG) << "Assemble aten input " << i << " " << DebugString(*inputs[i]) << " to "
                   << DebugString(input_holders[i]);
  }
  return Status::Success();
}

template <typename T>
Status MutiGearNpuGraphExecutor::UpdateInputsInner(const std::vector<const at::Tensor*> &inputs,
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
      if (!input_gears_[i].empty()) {
        TNG_RETURN_IF_ERROR(UpdateSpecificDims(input_holders[i], *inputs[i], input_gears_[i]));
      }
      TNG_RETURN_IF_ERROR(AssembleDataToGe(*inputs[i], input_holders[i]));
    } else {
      TNG_ASSERT(input_gears_[i].empty(), "CPU tensor unsupport set gears");
      if (host_input_holders_[i].second.first > 0) {
        TNG_RETURN_IF_ERROR(H2DMemcpy(host_input_holders_[i].first.data_ptr(), host_input_holders_[i].second.second,
                                      (*inputs[i]).data_ptr(), host_input_holders_[i].second.first, first_stream_));
      }
    }
    TNG_LOG(DEBUG) << "Update aten input " << i << " " << DebugString(*inputs[i]) << " to "
                   << DebugString(input_holders[i]);
  }
  return Status::Success();
}

Status MutiGearNpuGraphExecutor::Run(const std::vector<c10::optional<at::Tensor>> &torch_outputs,
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
    SetStageTime(ExecutorStage::kRunGraph);
    TNG_RETURN_IF_ERROR(
        Session::GetInstance().FastExecuteGraph(graph_data_->id, gert_inputs_holder_, gert_outputs_holder_, stream));

    TNG_RETURN_IF_ERROR(RefreshOutputShape(output_shapes_, gert_outputs_holder_));
  } else {
    TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs, data_ptrs, outputs_holder_, stream));
    SetStageTime(ExecutorStage::kAssembleOutputs);
    TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, outputs_holder_, stream));
    SetStageTime(ExecutorStage::kRunGraph);
    TNG_RETURN_IF_ERROR(RefreshOutputShape(output_shapes_, outputs_holder_));
  }

  outputs.clear();
  {
    RECORD_FUNCTION("RefreshAtTensorFromGearsGeTensor", {});
    for (size_t i = 0U; i < data_ptrs.size(); i++) {
      if (!torch_outputs.empty() && torch_outputs[i].has_value()) {
        outputs.push_back(torch_outputs[i].value());
        continue;
      }
      outputs.push_back(
          MakeAtTensor(output_shapes_[i], output_torch_dtype_[i], output_size_[i], std::move(data_ptrs[i])));
      TNG_LOG(DEBUG) << "Refresh gears torch output " << i << " " << DebugString(outputs[i]);
    }
  }
  TNG_LOG(EVENT) << "Muti gear executor call " << GenEventLog();
  TNG_LOG(INFO) << "Muti_gear npu graph executor run graph " << graph_data_->id << " on stream "
                << stream << " successfully.";
  if (fm_refreshable_) {
    TNG_ASSERT_NOTNULL(feature_map_block_);
    feature_map_block_->Free();
    feature_map_block_ = nullptr;
  }
  is_first_run_ = false;
  return Status::Success();
}
}  // namespace tng
