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

namespace tng {

Status GetCurrentStream(void **stream) {
  int device_index = -1;
  auto ret = aclrtGetDevice(&device_index);
  TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL get device failed, return %d", ret);
  *stream = c10_npu::getCurrentNPUStream(device_index).stream();
  return Status::Success();
}

Status AssembleStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  auto desc = ge_tensor.GetTensorDesc();

  if (tensor.device().is_privateuseone()) {
    desc.SetOriginShape(ge::Shape(tensor.sizes().vec()));
    desc.SetShape(ge::Shape(at_npu::native::get_npu_storage_sizes(tensor)));
  } else {
    desc.SetShape(ge::Shape(tensor.sizes().vec()));
  }

  TNG_ASSERT_GE_OK(ge_tensor.SetTensorDesc(desc));
  return Status::Success();
}

Status AssembleDataAndStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));
  TNG_RETURN_IF_ERROR(AssembleStorageShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AtNpuTensorToGeTensor(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  ge::TensorDesc desc;

  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  TNG_RETURN_IF_ERROR(AtDtypeToGeDtype(tensor.dtype().toScalarType(), ge_dtype));
  desc.SetDataType(ge_dtype);

  if (tensor.device().is_privateuseone()) {
    desc.SetPlacement(ge::Placement::kPlacementDevice);

    desc.SetOriginShape(ge::Shape(tensor.sizes().vec()));
    desc.SetOriginFormat(tensor.sizes().size() == 4 ? ge::Format::FORMAT_NCHW
                                                    : ge::Format::FORMAT_ND);

    // npu tensor may have internal format.
    const std::vector<int64_t> &storage_sizes = at_npu::native::get_npu_storage_sizes(tensor);
    desc.SetShape(ge::Shape(storage_sizes));
    desc.SetFormat(ge::Format(at_npu::native::get_npu_format(tensor)));

    TNG_LOG(DEBUG) << "Set ge tensor shape: " << storage_sizes << ", from npu at::tensor storage_sizes_.";
  } else {
    desc.SetPlacement(ge::Placement::kPlacementHost);

    desc.SetShape(ge::Shape(tensor.sizes().vec()));
    desc.SetFormat(ge::FORMAT_ND);

    TNG_LOG(DEBUG) << "Set ge tensor shape: " << tensor.sizes().vec() << ", from cpu at::tensor sizes.";
  }

  ge_tensor.SetTensorDesc(desc);
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));

  return Status::Success();
}

}  // namespace tng
