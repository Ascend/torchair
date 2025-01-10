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

template <typename T>
Status StaticNpuGraphExecutor::AssembleInputs(const std::vector<at::Tensor> &inputs, std::vector<T> &input_holders,
                                              void *stream) {
  RECORD_FUNCTION("AssembleInputs", std::vector<c10::IValue>({}));
  TNG_ASSERT(graph_data_->input_placements.size() == inputs.size());
  bool is_first_run = input_holders.empty();
  if (is_first_run) {
    input_holders.resize(inputs.size());
    host_input_holders_.resize(inputs.size());
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
        TNG_RETURN_IF_ERROR(AtTensorToGeTensor(inputs[i], input_holders[i]));
      } else {
        // In static graph input shape remains unchanged, only data ptr need to be updated.
        TNG_RETURN_IF_ERROR(AssembleDataToGe(inputs[i], input_holders[i]));
      }
      TNG_LOG(DEBUG) << "Assemble aten device input " << i << " " << DebugString(inputs[i]) << " to "
                     << DebugString(input_holders[i]);
    } else if (graph_data_->input_placements[i] == Placement::HOST) {
      if (!inputs[i].device().is_cpu()) {
        return Status::Error("Input %zu placement %s is incompatible with expected CPU.", i,
                             DebugString(inputs[i].device()).c_str());
      }
      TNG_LOG(DEBUG) << "Host input " << i << " " << DebugString(inputs[i]) << " need copy to device";
      TNG_RETURN_IF_ERROR(AssembleHostInputs(inputs[i], input_holders[i], host_input_holders_[i], stream, is_first_run));
      TNG_LOG(DEBUG) << "Assemble aten host input " << i << " " << DebugString(inputs[i]) << " to "
                     << DebugString(input_holders[i]);
    } else {
      TNG_ASSERT(false, "Invalid Placement::UNKNOWN of input %zu.", i);
    }
  }
  return Status::Success();
}

template <typename T>
Status StaticNpuGraphExecutor::AssembleHostInputs(const at::Tensor &input, T &input_holder,
                                                  std::pair<at::Tensor, std::pair<size_t, size_t>> &host_input_holder_,
                                                  void *stream, bool is_first_run) {
  if (is_first_run) {
    auto host_input_holder = at::empty(input.sizes(), input.options().device(at::kPrivateUse1));
    size_t dst_size = static_cast<size_t>(host_input_holder.numel() * host_input_holder.element_size());
    size_t src_size = static_cast<size_t>(input.numel() * input.element_size());
    auto copy_size = std::make_pair(src_size, dst_size);
    host_input_holder_ = std::make_pair(host_input_holder, copy_size);
    TNG_RETURN_IF_ERROR(AtTensorToGeTensor(host_input_holder_.first, input_holder));
    first_stream = stream;
  }
  if (host_input_holder_.second.first > 0) {
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
    auto ret = aclrtMemcpy(host_input_holder_.first.data_ptr(), host_input_holder_.second.second,
                           input.data_ptr(), host_input_holder_.second.first, ACL_MEMCPY_HOST_TO_DEVICE);
    TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL memory copy failed, return %d", ret);
  }
  return Status::Success();
}

template <typename T>
Status StaticNpuGraphExecutor::AssembleOutputs(const std::vector<c10::optional<at::Tensor>> &assigned_outputs,
                                               std::vector<at::Tensor> &outputs, std::vector<T> &output_holders) {
  RECORD_FUNCTION("AssembleOutputs", std::vector<c10::IValue>({}));
  bool is_first_run = output_holders.empty();
  if (is_first_run) {
    const std::vector<ge::DataType> &output_ge_dtypes = graph_data_->output_dtypes;
    std::vector<ge::Shape> output_ge_shapes;
    TNG_ASSERT_GE_OK(graph_data_->summary->GetOutputShapes(output_ge_shapes));
    TNG_ASSERT_EQ(output_ge_shapes.size(), output_ge_dtypes.size());

    output_holders.resize(output_ge_shapes.size());
    output_options_.resize(output_ge_shapes.size());
    output_shapes_.resize(output_ge_shapes.size());

    TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == output_shapes_.size());
    for (size_t i = 0U; i < output_ge_dtypes.size(); ++i) {
      if (assigned_outputs.empty() || !assigned_outputs[i].has_value()) {
        c10::ScalarType output_i_torch_dtype = c10::ScalarType::Float;
        TNG_RETURN_IF_ERROR(GeDtypeToAtDtype(output_ge_dtypes[i], output_i_torch_dtype));
        at::TensorOptions option = at::TensorOptions().dtype(output_i_torch_dtype).device(at::kPrivateUse1);
        output_options_[i] = option;
        output_shapes_[i] = output_ge_shapes[i].GetDims();
      }
    }
  }

  TNG_ASSERT(assigned_outputs.empty() || assigned_outputs.size() == output_holders.size());
  outputs.clear();
  for (size_t i = 0U; i < output_shapes_.size(); ++i) {
    if (!assigned_outputs.empty() && assigned_outputs[i].has_value()) {
      outputs.push_back(assigned_outputs[i].value());
      TNG_LOG(DEBUG) << "Assemble pre-assigned output " << i << " " << DebugString(outputs.back());
    } else {
      outputs.push_back(at::empty(output_shapes_[i], output_options_[i]));
      TNG_LOG(DEBUG) << "Create empty output " << i << " " << DebugString(outputs.back());
    }
    auto &output_i = outputs.back();
    if (is_first_run) {
      TNG_RETURN_IF_ERROR(AtTensorToGeTensor(output_i, output_holders[i]));
    } else {
      TNG_RETURN_IF_ERROR(AssembleDataToGe(output_i, output_holders[i]));
    }
    TNG_LOG(DEBUG) << "Assemble torch output " << i << " " << DebugString(output_i) << " to "
                   << DebugString(output_holders[i]);
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

Status StaticNpuGraphExecutor::Run(const std::vector<at::Tensor> &torch_inputs,
                                   const std::vector<c10::optional<at::Tensor>> &torch_outputs,
                                   std::vector<at::Tensor> &outputs, void *stream) {
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

    TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs, outputs, gert_outputs_holder_));

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
  } else {
    TNG_RETURN_IF_ERROR(AssembleInputs(torch_inputs, inputs_holder_, stream));

    TNG_RETURN_IF_ERROR(AssembleOutputs(torch_outputs, outputs, outputs_holder_));

    TNG_RETURN_IF_ERROR(Session::GetInstance().RunGraph(graph_data_->id, inputs_holder_, outputs_holder_, stream));
  }

  TNG_LOG(INFO) << "Static npu graph executor run run graph " << graph_data_->id << " on stream " << stream
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
