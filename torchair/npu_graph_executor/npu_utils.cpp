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
namespace {
ge::Format GetOriginFormatFromAtTensor(const at::Tensor &at_tensor) {
  size_t dim_num = at_tensor.sizes().size();
  ge::Format format = ge::FORMAT_ND;
  switch (dim_num) {
    case 4:
      format = ge::FORMAT_NCHW;
      break;
    case 5:
      format = ge::FORMAT_NCDHW;
      break;
    default:
      format = ge::FORMAT_ND;
  }

  return format;
}

inline bool IsBaseFormat(const ge::Format &format) {
  return (format == ge::FORMAT_ND) || (format == ge::FORMAT_NCHW) || (format == ge::FORMAT_NHWC) ||
         (format == ge::FORMAT_NCDHW);
}
}

Status GetCurrentStream(void **stream) {
  int device_index = -1;
  auto ret = aclrtGetDevice(&device_index);
  TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL get device failed, return %d", ret);
  *stream = c10_npu::getCurrentNPUStream(device_index).stream();
  return Status::Success();
}

Status AssembleDimsToOriginShape(const at::IntArrayRef &dims, ge::Tensor &ge_tensor){
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

Status AssembleDataAndStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
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
}  // namespace tng
