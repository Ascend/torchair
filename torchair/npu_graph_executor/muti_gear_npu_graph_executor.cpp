#include "external/graph/types.h"
#include "muti_gear_npu_graph_executor.h"

#include <ATen/record_function.h>

#include <utility>
#include "checker.h"
#include "logger.h"
#include "session.h"
#include "torch/torch.h"
#include "utils.h"
#include "npu_utils.h"
#include "AllocatorManager.h"
#include "acl/acl_rt.h"

namespace tng {
namespace {
void FreeMemBlock(void *data) {
  if (data != nullptr) {
    static_cast<ge::MemBlock *>(data)->Free();
    data = nullptr;
  }
}

at::Tensor MakeAtTensor(ge::Tensor &ge_tensor, size_t tensor_nbytes, c10::ScalarType &torch_dtype, ge::MemBlock *block) {
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
      std::move(storage), c10::DispatchKeySet{c10::DispatchKey::PrivateUse1, c10::DispatchKey::AutogradPrivateUse1},
      c10::scalarTypeToTypeMeta(torch_dtype));
  const std::vector<int64_t> &dims = GetGeTensorShape(ge_tensor);
  at::TensorImpl *tensor_impl = tensor.unsafeGetTensorImpl();
  tensor_impl->set_sizes_contiguous(dims);
  return tensor;
}

Status ParseInputGears(const std::vector<at::Tensor> &inputs, std::vector<std::vector<int64_t>> &input_gears,
                       const std::vector<std::vector<int64_t>> &inputs_shape) {
  if (inputs_shape.empty()) {
    for (const auto & input : inputs) {
      input_gears.emplace_back(input.dim(), -1);
    }
    TNG_LOG(DEBUG) << "Parse input_gears but inputs_shape is empty, so construct it as "
                   << DebugString(input_gears);
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
        gear_dims.push_back(static_cast<int64_t >(dim_index));
      }
    }
    input_gears.emplace_back(gear_dims);
  }

  TNG_LOG(DEBUG) << "Parse input_gears " << DebugString(input_gears) << " from input shape "
                 <<  DebugString(inputs_shape);
  return Status::Success();
}
}  // namespace

Status MutiGearNpuGraphExecutor::AssembleInputs(const std::vector<at::Tensor> &inputs, void *stream) {
  TNG_ASSERT(graph_data_->input_placements.size() == inputs.size());
  bool is_first_run = inputs_holder_.empty();
  if (is_first_run) {
    inputs_holder_.resize(inputs.size());
    host_input_holders_.resize(inputs.size());
    TNG_RETURN_IF_ERROR(ParseInputGears(inputs, input_gears_, graph_data_->inputs_shape));
  }
  for (size_t i = 0U; i < inputs.size(); ++i) {
    if (graph_data_->input_placements[i] == Placement::DEVICE) {
      if (!inputs[i].device().is_privateuseone()) {
        return Status::Error("Input %zu placement %s is incompatible with expected PrivateUse1.", i,
                             DebugString(inputs[i].device()).c_str());
      }
      if (is_first_run) {
        TNG_RETURN_IF_ERROR(AtNpuTensorToGeTensor(inputs[i], inputs_holder_[i]));
        if (!input_gears_[i].empty()) {
          TNG_ASSERT(IsBaseFormat(inputs_holder_[i].GetFormat()),
                     "Gear input expect format is base format not private format, but got format is %d.",
                     static_cast<int32_t>(inputs_holder_[i].GetFormat()));
        }
      } else {
        if (!input_gears_[i].empty()) {
          // 只刷新特定维度的shape to ge tensor
          auto torch_size = inputs[i].sizes();
          for (auto dim : input_gears_[i]) {
            auto real_dim_shape = torch_size[dim];
            TNG_ASSERT_GE_OK(inputs_holder_[i].SetShapeDim(dim, real_dim_shape));
          }
        }
        TNG_RETURN_IF_ERROR(AssembleDataToGe(inputs[i], inputs_holder_[i]));
      }
      TNG_LOG(DEBUG) << "Assemble aten device input " << i << " " << DebugString(inputs[i]) << " to "
                     << DebugString(inputs_holder_[i]);
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
        TNG_RETURN_IF_ERROR(AtTensorToGeTensor(host_input_holders_[i].first, inputs_holder_[i]));
      }
      auto stream_ret = aclrtSynchronizeStream(stream);
      TNG_ASSERT(stream_ret == ACL_ERROR_NONE, "ACL sync stream failed, return %d", stream_ret);
      auto ret = aclrtMemcpy(host_input_holders_[i].first.data_ptr(), host_input_holders_[i].second,
                             inputs[i].data_ptr(), host_input_holders_[i].second, ACL_MEMCPY_HOST_TO_DEVICE);
      TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL memory copy failed, return %d", ret);
      TNG_LOG(DEBUG) << "Assemble aten host input " << i << " " << DebugString(inputs[i]) << " to "
                     << DebugString(inputs_holder_[i]);
    } else {
      TNG_ASSERT(false, "Invalid Placement::UNKNOWN of input %zu.", i);
    }
  }

  return Status::Success();
}

Status MutiGearNpuGraphExecutor::AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                                                 std::vector<ge::MemBlock *> &output_mem_block, void *stream) {
  bool is_first_run = outputs_holder_.empty();
  if (is_first_run) {
    auto output_ge_dtypes = graph_data_->output_dtypes;
    TNG_ASSERT_GE_OK(graph_data_->summary->GetOutputShapes(output_shapes_));
    TNG_ASSERT_EQ(output_shapes_.size(), output_ge_dtypes.size());
    outputs_holder_.resize(output_shapes_.size());
    output_options_.resize(output_shapes_.size());
    output_size_.resize(output_shapes_.size());
    output_torch_dtype_.resize(output_shapes_.size());

    TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == output_shapes_.size());
    for (size_t i = 0U; i < output_ge_dtypes.size(); ++i) {
      TNG_RETURN_IF_ERROR(GeDtypeToAtDtype(output_ge_dtypes[i], output_torch_dtype_[i]));
      outputs_holder_[i].SetPlacement(ge::Placement::kPlacementDevice);
      outputs_holder_[i].SetDataType(output_ge_dtypes[i]);
      outputs_holder_[i].SetFormat(ge::FORMAT_ND);
      TNG_ASSERT_GE_OK(outputs_holder_[i].SetShapeDimNum(output_shapes_[i].GetDimNum()));
      for (size_t index = 0U; index < output_shapes_[i].GetDimNum(); ++index) {
        TNG_ASSERT_GE_OK(outputs_holder_[i].SetShapeDim(index, output_shapes_[i].GetDim(index)));
      }
      output_size_[i] = at::detail::computeStorageNbytesContiguous(output_shapes_[i].GetShapeSize(),
                                                                   elementSize(output_torch_dtype_[i]));
    }
  }
  output_mem_block.resize(output_shapes_.size());
  TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == outputs_holder_.size());
  std::shared_ptr<ge::Allocator> allocator; // allocator reuse in once execute
  for (size_t i = 0U; i < output_shapes_.size(); ++i) {
    if (!assigned_outputs.empty() && assigned_outputs[i].has_value()) {
      if (is_first_run) {
        TNG_RETURN_IF_ERROR(AtNpuTensorToGeTensor(assigned_outputs[i].value(), outputs_holder_[i]));
      } else {
        TNG_RETURN_IF_ERROR(AssembleDataAndStorageShapeToGe(assigned_outputs[i].value(), outputs_holder_[i]));
      }
      TNG_LOG(DEBUG) << "Assemble pre-assigned output " << i << " " << DebugString(assigned_outputs[i].value())
                     << " to " << DebugString(outputs_holder_[i]);
      continue;
    }

    if (allocator == nullptr) {
      // Register allocator for GE before run, according to stream.
      allocator = AllocatorManager::GetInstance().EnsureAllocatorRegistered(stream);
      TNG_ASSERT_NOTNULL(allocator);
    }

    ge::MemBlock *block = allocator->Malloc(output_size_[i]);
    TNG_ASSERT_NOTNULL(block);
    output_mem_block[i] = block;
    const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t *data) {};
    outputs_holder_[i].ResetData(static_cast<uint8_t *>(block->GetAddr()),
                                 static_cast<size_t>(output_size_[i]), kDoNothing);
    TNG_LOG(DEBUG) << "Malloc " << output_size_[i] << " for ge gear output tensor.";
  }

  return Status::Success();
}

Status MutiGearNpuGraphExecutor::Run(const std::vector<at::Tensor> &torch_inputs,
                                     const std::vector<c10::optional<at::Tensor>> &torch_outputs,
                                     std::vector<at::Tensor> &outputs, void *stream) {
  std::vector<ge::MemBlock *> output_mem_block;
  if (stream == nullptr) {
    TNG_RETURN_IF_ERROR(GetCurrentStream(&stream));
  }
  {
    RECORD_FUNCTION("AssembleInputs", std::vector<c10::IValue>({}));
    TNG_RETURN_IF_ERROR(AssembleInputs(torch_inputs, stream));
  }
  {
    RECORD_FUNCTION("AssembleOutputs", std::vector<c10::IValue>({}));
    TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs, output_mem_block, stream));
  }

  if (is_first_run_) {
    TNG_ASSERT_GE_OK(graph_data_->summary->GetFeatureMemoryBaseRefreshable(fm_refreshable_));
    TNG_RETURN_IF_ERROR(AllocAndSetConstMemory(stream));
    is_first_run_ = false;
  }

  TNG_RETURN_IF_ERROR(AllocAndUpdateFeatureMemory(stream));

  {
    RECORD_FUNCTION("RunGraphWithStreamAsync", std::vector<c10::IValue>({}));
    TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, outputs_holder_, stream));
  }
  outputs.clear();
  {
    RECORD_FUNCTION("RefreshAtTensorFromGearsGeTensor", {});
    for (size_t i = 0U; i < output_mem_block.size(); i++) {
      if (!torch_outputs.empty() && torch_outputs[i].has_value()) {
        outputs.push_back(torch_outputs[i].value());
        continue;
      }
      outputs.push_back(MakeAtTensor(outputs_holder_[i], output_size_[i],
                                     output_torch_dtype_[i], output_mem_block[i]));
      TNG_LOG(DEBUG) << "Refresh gears torch output " << i << " " << DebugString(outputs[i])
                      << " from ge output " << DebugString(outputs_holder_[i]);
    }
  }
  TNG_LOG(INFO) << "MutiGearNpuGraphExecutor::Run graph " << graph_data_->id << " on stream " << stream
                << " successfully.";
  if (fm_refreshable_) {
    TNG_ASSERT_NOTNULL(feature_map_block_);
    feature_map_block_->Free();
    feature_map_block_ = nullptr;
  }
  return Status::Success();
}
}  // namespace tng
