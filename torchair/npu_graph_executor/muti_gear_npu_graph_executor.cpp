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
void FreeMemBlock(void *data) {
  if (data != nullptr) {
    static_cast<ge::MemBlock *>(data)->Free();
    data = nullptr;
  }
}

at::Tensor MakeAtTensor(const std::vector<int64_t> &dims, c10::ScalarType &torch_dtype, size_t tensor_nbytes,
                        ge::MemBlock *block) {
  static torch::DeleterFnPtr kFreeMemBlock = &FreeMemBlock;
  at::DataPtr c10_data_ptr(block->GetAddr(), block, kFreeMemBlock, c10::DeviceType::PrivateUse1);
  at::Storage storage;
  // get npu storage constructor from register and construct storage
  auto fptr = c10::GetStorageImplCreate(c10::DeviceType::PrivateUse1);
  auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
#if defined(TNG_TORCH_VERSION) && (TNG_TORCH_VERSION < 20300)  // v2.3.0
  storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator, true);
#else
  storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator->allocate(0), allocator, true);
#endif
  storage.unsafeGetStorageImpl()->set_nbytes(tensor_nbytes);
  storage.set_data_ptr(std::move(c10_data_ptr));

  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet{c10::DispatchKey::PrivateUse1, c10::DispatchKey::AutogradPrivateUse1},
      c10::scalarTypeToTypeMeta(torch_dtype));
  at::TensorImpl *tensor_impl = tensor.unsafeGetTensorImpl();
  tensor_impl->set_sizes_contiguous(dims);
  return tensor;
}

Status ParseInputGears(const std::vector<at::Tensor> &inputs, std::vector<std::vector<int64_t>> &input_gears,
                       const std::vector<std::vector<int64_t>> &inputs_shape) {
  if (inputs_shape.empty()) {
    for (const auto &input : inputs) {
      input_gears.emplace_back(input.dim(), -1);
    }
    TNG_LOG(DEBUG) << "Parse input_gears but inputs_shape is empty, so construct it as " << DebugString(input_gears);
    return Status::Success();
  }
  for (size_t i = 0U; i < inputs.size(); ++i) {
    if (inputs_shape[i].size() == 1 && inputs_shape[i][0] == UNKONWN_DIM_NUM) {
      input_gears.emplace_back(inputs[i].dim(), -1);
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

inline Status UpdateTensorInfos(ge::Tensor &ge_tensor, const std::vector<int64_t> &shape, const ge::Format format,
                                const ge::DataType data_type) {
  TNG_ASSERT_GE_OK(ge_tensor.SetDataType(data_type));
  TNG_ASSERT_GE_OK(ge_tensor.SetPlacement(ge::Placement::kPlacementDevice));
  TNG_ASSERT_GE_OK(ge_tensor.SetFormat(format));
  TNG_ASSERT_GE_OK(ge_tensor.SetShapeDimNum(shape.size()));
  for (size_t index = 0U; index < shape.size(); ++index) {
    TNG_ASSERT_GE_OK(ge_tensor.SetShapeDim(index, shape[index]));
  }
  return Status::Success();
}

inline Status UpdateTensorInfos(gert::Tensor &ge_tensor, const std::vector<int64_t> &shape, const ge::Format format,
                                const ge::DataType data_type) {
  ge_tensor.SetDataType(data_type);
  ge_tensor.SetPlacement(gert::TensorPlacement::kOnDeviceHbm);
  ge_tensor.SetOriginFormat(format);
  ge_tensor.SetStorageFormat(format);
  TNG_RETURN_IF_ERROR(AssembleDimsToShape(shape, ge_tensor));
  return Status::Success();
}

inline Status UpdateTensorData(ge::Tensor &ge_tensor, void *addr, const size_t data_size) {
  const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t *data) {};
  TNG_ASSERT_GE_OK(ge_tensor.ResetData(static_cast<uint8_t *>(addr), static_cast<size_t>(data_size), kDoNothing));
  return Status::Success();
}

inline Status UpdateTensorData(gert::Tensor &ge_tensor, void *addr, const size_t data_size) {
  TNG_ASSERT_GE_OK(ge_tensor.MutableTensorData().SetAddr(addr, nullptr));
  ge_tensor.MutableTensorData().SetSize(data_size);
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

template <typename T>
Status MutiGearNpuGraphExecutor::AssembleInputs(const std::vector<at::Tensor> &inputs, std::vector<T> &input_holders,
                                                void *stream) {
  RECORD_FUNCTION("AssembleInputs", std::vector<c10::IValue>({}));
  TNG_ASSERT(graph_data_->input_placements.size() == inputs.size());
  bool is_first_run = input_holders.empty();
  if (is_first_run) {
    input_holders.resize(inputs.size());
    host_input_holders_.resize(inputs.size());
    TNG_RETURN_IF_ERROR(ParseInputGears(inputs, input_gears_, graph_data_->inputs_shape));
    TNG_ASSERT(graph_data_->frozen_input_flag_list.size() == inputs.size());
  }
  for (size_t i = 0U; i < inputs.size(); ++i) {
    if (!is_first_run && graph_data_->frozen_input_flag_list[i]) {
      TNG_LOG(DEBUG) << "Assemble frozen input " << i << " " << DebugString(inputs[i])
                     << ", does not need to assemble if not in first run.";
      continue;
    }
    if (graph_data_->input_placements[i] == Placement::DEVICE) {
      if (!inputs[i].device().is_privateuseone()) {
        return Status::Error("Input %zu placement %s is incompatible with expected PrivateUse1.", i,
                             DebugString(inputs[i].device()).c_str());
      }
      if (is_first_run) {
        if (!input_gears_[i].empty()) {
          TNG_ASSERT(IsBaseFormat(ge::Format(at_npu::native::get_npu_format(inputs[i]))),
                     "Gear input expect format is base format not private format, but got format is %s.",
                     ge::GetFormatName(ge::Format(at_npu::native::get_npu_format(inputs[i]))));
        }
        TNG_RETURN_IF_ERROR(AtNpuTensorToGeTensor(inputs[i], input_holders[i]));
      } else {
        if (!input_gears_[i].empty()) {
          TNG_RETURN_IF_ERROR(UpdateSpecificDims(input_holders[i], inputs[i], input_gears_[i]));
        }
        TNG_RETURN_IF_ERROR(AssembleDataToGe(inputs[i], input_holders[i]));
      }
      TNG_LOG(DEBUG) << "Assemble aten device input " << i << " " << DebugString(inputs[i])
                     << " to " << DebugString(input_holders[i]);
    } else if (graph_data_->input_placements[i] == Placement::HOST) {
      if (!inputs[i].device().is_cpu()) {
        return Status::Error("Input %zu placement %s is incompatible with expected CPU.", i,
                             DebugString(inputs[i].device()).c_str());
      }
      TNG_ASSERT(input_gears_[i].empty(), "CPU tensor unsupport set gears");
      TNG_LOG(DEBUG) << "Host input " << i << " " << DebugString(inputs[i]) << " need copy to device";

      if (is_first_run) {
        auto host_input_holder = at::empty(inputs[i].sizes(), inputs[i].options().device(at::kPrivateUse1));
        size_t copy_size = static_cast<size_t>(inputs[i].numel() * inputs[i].element_size());
        host_input_holders_[i] = std::make_pair(host_input_holder, copy_size);
        TNG_RETURN_IF_ERROR(AtTensorToGeTensor(host_input_holders_[i].first, input_holders[i]));
      }
      if (host_input_holders_[i].second > 0) {
        auto stream_ret = aclrtSynchronizeStream(stream);
        TNG_ASSERT(stream_ret == ACL_ERROR_NONE, "ACL sync stream failed, return %d", stream_ret);
        auto ret = aclrtMemcpy(host_input_holders_[i].first.data_ptr(), host_input_holders_[i].second,
                               inputs[i].data_ptr(), host_input_holders_[i].second, ACL_MEMCPY_HOST_TO_DEVICE);
        TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL memory copy failed, return %d", ret);
      }
      TNG_LOG(DEBUG) << "Assemble aten host input " << i << " " << DebugString(inputs[i])
                     << " to " << DebugString(input_holders[i]);
    } else {
      TNG_ASSERT(false, "Invalid Placement::UNKNOWN of input %zu.", i);
    }
  }

  return Status::Success();
}

template <typename T>
Status MutiGearNpuGraphExecutor::AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                                                 std::vector<ge::MemBlock *> &output_mem_blocks,
                                                 std::vector<T> &output_holders, void *stream) {
  RECORD_FUNCTION("AssembleOutputs", std::vector<c10::IValue>({}));
  bool is_first_run = output_holders.empty();
  if (is_first_run) {
    auto output_ge_dtypes = graph_data_->output_dtypes;
    std::vector<ge::Shape> output_ge_shapes;
    TNG_ASSERT_GE_OK(graph_data_->summary->GetOutputShapes(output_ge_shapes));
    TNG_ASSERT_EQ(output_ge_shapes.size(), output_ge_dtypes.size());
    output_holders.resize(output_ge_dtypes.size());
    output_size_.resize(output_ge_dtypes.size());
    output_shapes_.resize(output_ge_dtypes.size());
    output_torch_dtype_.resize(output_ge_dtypes.size());
    real_output_shape_.resize(output_ge_dtypes.size());

    TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == output_ge_dtypes.size());
    for (size_t i = 0U; i < output_ge_dtypes.size(); ++i) {
      TNG_RETURN_IF_ERROR(GeDtypeToAtDtype(output_ge_dtypes[i], output_torch_dtype_[i]));
      output_shapes_[i] = output_ge_shapes[i].GetDims();
      real_output_shape_[i] = output_ge_shapes[i].GetDims();
      output_size_[i] =
          at::detail::computeStorageNbytesContiguous(output_shapes_[i], elementSize(output_torch_dtype_[i]));

      TNG_RETURN_IF_ERROR(UpdateTensorInfos(output_holders[i], output_shapes_[i], ge::FORMAT_ND, output_ge_dtypes[i]));
    }
  }

  TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == output_holders.size());
  const std::shared_ptr<ge::Allocator> &allocator = AllocatorManager::GetInstance().EnsureAllocatorRegistered(stream);
  TNG_ASSERT_NOTNULL(allocator);
  output_mem_blocks.resize(output_holders.size());
  for (size_t i = 0U; i < output_holders.size(); ++i) {
    if (!assigned_outputs.empty() && assigned_outputs[i].has_value()) {
      if (is_first_run) {
        TNG_RETURN_IF_ERROR(AtNpuTensorToGeTensor(assigned_outputs[i].value(), output_holders[i]));
      } else {
        TNG_RETURN_IF_ERROR(AssembleDataAndStorageShapeToGe(assigned_outputs[i].value(), output_holders[i]));
      }
      TNG_LOG(DEBUG) << "Assemble pre-assigned output " << i << " " << DebugString(assigned_outputs[i].value())
                     << " to " << DebugString(output_holders[i]);
      continue;
    }

    ge::MemBlock *block = allocator->Malloc(output_size_[i]);
    TNG_ASSERT_NOTNULL(block);
    output_mem_blocks[i] = block;
    TNG_RETURN_IF_ERROR(UpdateTensorData(output_holders[i], block->GetAddr(), output_size_[i]));
    TNG_LOG(DEBUG) << "Malloc " << output_size_[i] << " for ge gear output tensor.";
  }

  return Status::Success();
}

Status MutiGearNpuGraphExecutor::Run(const std::vector<at::Tensor> &torch_inputs,
                                     const std::vector<c10::optional<at::Tensor>> &torch_outputs,
                                     std::vector<at::Tensor> &outputs, void *stream) {
  std::vector<ge::MemBlock *> output_mem_blocks;
  if (stream == nullptr) {
    TNG_RETURN_IF_ERROR(GetCurrentStream(&stream));
  }

  if (is_first_run_) {
    TNG_ASSERT_GE_OK(graph_data_->summary->GetFeatureMemoryBaseRefreshable(fm_refreshable_));
    TNG_RETURN_IF_ERROR(AllocAndSetConstMemory(stream));
  }
  TNG_RETURN_IF_ERROR(AllocAndUpdateFeatureMemory(stream));

  static bool enable_load_execute_graph =
      Session::GetInstance().IsFastLoadGraphSupported() && Session::GetInstance().IsFastExecuteGraphSupported();
  if (enable_load_execute_graph) {
    TNG_RETURN_IF_ERROR(AssembleInputs(torch_inputs, gert_inputs_holder_, stream));

    TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs, output_mem_blocks, gert_outputs_holder_, stream));

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

    TNG_RETURN_IF_ERROR(RefreshOutputShape(real_output_shape_, gert_outputs_holder_));
  } else {
    TNG_RETURN_IF_ERROR(AssembleInputs(torch_inputs, inputs_holder_, stream));

    TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs, output_mem_blocks, outputs_holder_, stream));

    TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, outputs_holder_, stream));

    TNG_RETURN_IF_ERROR(RefreshOutputShape(real_output_shape_, outputs_holder_));
  }

  outputs.clear();
  {
    RECORD_FUNCTION("RefreshAtTensorFromGearsGeTensor", {});
    for (size_t i = 0U; i < output_mem_blocks.size(); i++) {
      if (!torch_outputs.empty() && torch_outputs[i].has_value()) {
        outputs.push_back(torch_outputs[i].value());
        continue;
      }
      outputs.push_back(
          MakeAtTensor(real_output_shape_[i], output_torch_dtype_[i], output_size_[i], output_mem_blocks[i]));
      TNG_LOG(DEBUG) << "Refresh gears torch output " << i << " " << DebugString(outputs[i]);
    }
  }
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
