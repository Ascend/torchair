#include "npu_utils.h"
#include "checker.h"
#include "graph_data.h"
#include "logger.h"
#include "utils.h"

#include "torch_npu/inc/core/NPUFormat.h"
#include "torch_npu/inc/core/NPUStream.h"

#include "acl/acl_rt.h"
#include "external/graph/types.h"

namespace tng {
static constexpr size_t kFormatNchwRank = 4;
static constexpr size_t kFormatNcdhwRank = 5;

namespace {
ge::Format GetOriginFormatFromAtTensor(const at::Tensor &at_tensor) {
  size_t dim_num = at_tensor.sizes().size();
  ge::Format format = ge::FORMAT_ND;
  switch (dim_num) {
    case kFormatNchwRank:
      format = ge::FORMAT_NCHW;
      break;
    case kFormatNcdhwRank:
      format = ge::FORMAT_NCDHW;
      break;
    default:
      format = ge::FORMAT_ND;
  }

  return format;
}
}  // namespace

bool IsBaseFormat(const ge::Format &format) {
  return (format == ge::FORMAT_ND) || (format == ge::FORMAT_NCHW) || (format == ge::FORMAT_NHWC) ||
         (format == ge::FORMAT_NCDHW);
}

Status GetCurrentStream(void **stream) {
  int device_index = -1;
  auto ret = aclrtGetDevice(&device_index);
  TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL get device failed, return %d", ret);
  *stream = c10_npu::getCurrentNPUStream(device_index).stream();
  return Status::Success();
}

Status AssembleDimsToOriginShape(const at::IntArrayRef &dims, ge::Tensor &ge_tensor) {
  if (ge_tensor.GetOriginShapeDimNum() != dims.size()) {
    TNG_ASSERT_GE_OK(ge_tensor.SetOriginShapeDimNum(dims.size()));
  }
  for (size_t i = 0U; i < dims.size(); ++i) {
    TNG_ASSERT_GE_OK(ge_tensor.SetOriginShapeDim(i, dims[i]));
  }

  return Status::Success();
}

Status AssembleStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  if (tensor.device().is_privateuseone()) {
    const bool is_base_format = IsBaseFormat(ge_tensor.GetFormat());
    TNG_ASSERT(is_base_format || (tensor.storage_offset() == 0),
               "Invalid at::tensor with internal format and offset is %lld.", tensor.storage_offset());

    if (is_base_format) {
      TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), ge_tensor));
    } else {
      TNG_RETURN_IF_ERROR(AssembleDimsToShape(at_npu::native::get_npu_storage_sizes(tensor), ge_tensor));
    }
    TNG_RETURN_IF_ERROR(AssembleDimsToOriginShape(tensor.sizes(), ge_tensor));
  } else {
    TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), ge_tensor));
  }

  return Status::Success();
}

Status AssembleStorageShapeToGe(const at::Tensor &tensor, gert::Tensor &ge_tensor) {
  if (tensor.device().is_privateuseone()) {
    const bool is_base_format = IsBaseFormat(ge_tensor.GetStorageFormat());
    TNG_ASSERT(is_base_format || (tensor.storage_offset() == 0),
               "Invalid at::tensor with internal format and offset is %lld.", tensor.storage_offset());

    if (!is_base_format) {
      const std::vector<int64_t> &storage_sizes = at_npu::native::get_npu_storage_sizes(tensor);
      TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), storage_sizes, ge_tensor));
      const int64_t num =
          std::accumulate(storage_sizes.cbegin(), storage_sizes.cend(), 1LL, std::multiplies<int64_t>());
      ge_tensor.MutableTensorData().SetSize(static_cast<size_t>(num * tensor.element_size()));
      return Status::Success();
    }
  }

  TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), ge_tensor));
  ge_tensor.MutableTensorData().SetSize(static_cast<size_t>(tensor.numel() * tensor.element_size()));
  return Status::Success();
}

Status AssembleDataAndStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));
  TNG_RETURN_IF_ERROR(AssembleStorageShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AssembleDataAndStorageShapeToGe(const at::Tensor &tensor, gert::Tensor &ge_tensor) {
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));
  TNG_RETURN_IF_ERROR(AssembleStorageShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AtNpuTensorToGeTensor(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  ge::TensorDesc desc;

  if (tensor.device().is_privateuseone()) {
    desc.SetOriginFormat(GetOriginFormatFromAtTensor(tensor));
    desc.SetOriginShape(ge::Shape(tensor.sizes().vec()));
    TNG_LOG(DEBUG) << "Set ge tensor origin shape: " << tensor.sizes() << ", from npu at::tensor shape.";

    desc.SetPlacement(ge::Placement::kPlacementDevice);
    desc.SetFormat(ge::Format(at_npu::native::get_npu_format(tensor)));
  } else {
    desc.SetPlacement(ge::Placement::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
  }

  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  TNG_RETURN_IF_ERROR(AtDtypeToGeDtype(tensor.dtype().toScalarType(), ge_dtype));
  desc.SetDataType(ge_dtype);

  TNG_ASSERT_GE_OK(ge_tensor.SetTensorDesc(desc));
  TNG_RETURN_IF_ERROR(AssembleDataAndStorageShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AtNpuTensorToGeTensor(const at::Tensor &tensor, gert::Tensor &ge_tensor) {
  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  TNG_RETURN_IF_ERROR(AtDtypeToGeDtype(tensor.dtype().toScalarType(), ge_dtype));
  ge_tensor.SetDataType(ge_dtype);

  if (tensor.device().is_privateuseone()) {
    ge_tensor.SetPlacement(gert::TensorPlacement::kOnDeviceHbm);
    ge_tensor.SetOriginFormat(GetOriginFormatFromAtTensor(tensor));
    ge_tensor.SetStorageFormat(ge::Format(at_npu::native::get_npu_format(tensor)));
  } else {
    ge_tensor.SetPlacement(gert::TensorPlacement::kOnHost);
    ge_tensor.SetOriginFormat(ge::FORMAT_ND);
    ge_tensor.SetStorageFormat(ge::FORMAT_ND);
  }
  TNG_RETURN_IF_ERROR(AssembleDataAndStorageShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AssembleDimsToShape(const at::IntArrayRef &origin_dims, const at::IntArrayRef &storage_dims,
                           gert::Tensor &ge_tensor) {
  if (ge_tensor.GetOriginShape().GetDimNum() != origin_dims.size()) {
    ge_tensor.MutableOriginShape().SetDimNum(origin_dims.size());
  }
  if (ge_tensor.GetStorageShape().GetDimNum() != storage_dims.size()) {
    ge_tensor.MutableStorageShape().SetDimNum(storage_dims.size());
  }
  for (size_t i = 0U; i < origin_dims.size(); ++i) {
    ge_tensor.MutableOriginShape().SetDim(i, origin_dims[i]);
  }
  for (size_t i = 0U; i < storage_dims.size(); ++i) {
    ge_tensor.MutableStorageShape().SetDim(i, storage_dims[i]);
  }

  return Status::Success();
}

Status AssembleFrozenOption(const std::vector <bool> &frozen_input_flag_list,
                            const std::vector <at::Tensor> &torch_inputs,
                            std::string &frozen_option_value) {
  if (frozen_input_flag_list.empty()) {
    return Status::Success();
  }
  TNG_ASSERT(frozen_input_flag_list.size() == torch_inputs.size());
  std::stringstream frozen_input_flag_list_stream;
  for (size_t i = 0U; i < frozen_input_flag_list.size(); i++) {
    if (frozen_input_flag_list[i]) {
      if (frozen_input_flag_list_stream.str() != "") {
        frozen_input_flag_list_stream << ";";
      }
      const std::vector<int64_t> &storage_sizes = at_npu::native::get_npu_storage_sizes(torch_inputs[i]);
      const int64_t num = std::accumulate(storage_sizes.cbegin(), storage_sizes.cend(), 1LL,
                                          std::multiplies<int64_t>());
      frozen_input_flag_list_stream << i << "," << reinterpret_cast<uintptr_t>(torch_inputs[i].data_ptr()) << ","
                                    << static_cast<size_t>(num * torch_inputs[i].element_size());
    }
  }
  frozen_option_value = frozen_input_flag_list_stream.str();
  return Status::Success();
}

Status GetShapeFromGeTensor(std::vector<int64_t> &real_output_shape, const ge::Tensor &ge_tensor) {
  TNG_ASSERT_EQ(real_output_shape.size(), ge_tensor.GetShapeDimNum());
  for (size_t i = 0U; i < real_output_shape.size(); ++i) {
    real_output_shape[i] = ge_tensor.GetShapeDim(i);
  }
  return Status::Success();
}

Status GetShapeFromGeTensor(std::vector<int64_t> &real_output_shape, const gert::Tensor &ge_tensor) {
  TNG_ASSERT_EQ(real_output_shape.size(), ge_tensor.GetStorageShape().GetDimNum());
  for (size_t i = 0U; i < real_output_shape.size(); ++i) {
    real_output_shape[i] = ge_tensor.GetStorageShape().GetDim(i);
  }
  return Status::Success();
}

}  // namespace tng
