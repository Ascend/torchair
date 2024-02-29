#include "npu_utils.h"
#include "checker.h"
#include "graph_data.h"
#include "logger.h"
#include "utils.h"

#include "torch_npu/inc/core/NPUFormat.h"
#include "torch_npu/inc/core/NPUStream.h"

#include "acl/acl_rt.h"
#include "external/graph/types.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/tensor_adapter.h"

namespace tng {

Status GetCurrentStream(void **stream) {
  int device_index = -1;
  auto ret = aclrtGetDevice(&device_index);
  TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL get device failed, return %d", ret);
  *stream = c10_npu::getCurrentNPUStream(device_index).stream();
  return Status::Success();
}

Status AssembleStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  ge::GeTensorDesc &desc = ge::TensorAdapter::AsGeTensorShared(ge_tensor).MutableTensorDesc();

  if (tensor.device().is_privateuseone()) {
    AssembleDimsToGeShape(tensor.sizes(), desc.MutableOriginShape());

    const bool is_base_format = (desc.GetFormat() == ge::FORMAT_ND) || (desc.GetFormat() == ge::FORMAT_NCHW) ||
                                (desc.GetFormat() == ge::FORMAT_NHWC);
    TNG_ASSERT(is_base_format || (tensor.storage_offset() == 0),
               "Invalid at::tensor with internal format and offset is %lld.", tensor.storage_offset());
    const std::vector<int64_t> &dims = is_base_format ? tensor.sizes().vec() :
                                                        at_npu::native::get_npu_storage_sizes(tensor);
    AssembleDimsToGeShape(dims, desc.MutableShape());
  } else {
    AssembleDimsToGeShape(tensor.sizes(), desc.MutableShape());
  }

  return Status::Success();
}

Status AssembleDataAndStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));
  TNG_RETURN_IF_ERROR(AssembleStorageShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AtNpuTensorToGeTensor(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  ge::GeTensorDesc &desc = ge::TensorAdapter::AsGeTensorShared(ge_tensor).MutableTensorDesc();

  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  TNG_RETURN_IF_ERROR(AtDtypeToGeDtype(tensor.dtype().toScalarType(), ge_dtype));
  desc.SetDataType(ge_dtype);

  if (tensor.device().is_privateuseone()) {
    desc.SetPlacement(ge::Placement::kPlacementDevice);
    desc.SetOriginShape(ge::GeShape(tensor.sizes().vec()));
    desc.SetOriginFormat(tensor.sizes().size() == 4 ? ge::Format::FORMAT_NCHW
                                                    : ge::Format::FORMAT_ND);
    desc.SetFormat(ge::Format(at_npu::native::get_npu_format(tensor)));
  } else {
    desc.SetPlacement(ge::Placement::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
  }

  TNG_RETURN_IF_ERROR(AssembleDataAndStorageShapeToGe(tensor, ge_tensor));

  return Status::Success();
}

}  // namespace tng
