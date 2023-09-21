#include "npu_utils.h"
#include "external/graph/types.h"
#include "graph/tensor.h"
#include "graph/utils/type_utils.h"

#include "checker.h"
#include "logger.h"
#include "torch/torch.h"

#include "acl/acl_rt.h"
#include "torch_npu/inc/core/NPUStream.h"

namespace tng {

Status GetCurrentStream(void **stream) {
  int device_index = -1;
  auto ret = aclrtGetDevice(&device_index);
  TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL get device failed, return %d", ret);
  *stream = c10_npu::getCurrentNPUStream(device_index).stream();
  return Status::Success();
}

}  // namespace tng
