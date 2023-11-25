#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

aclError aclrtSetDevice(int32_t deviceId) {
  return ACL_ERROR_NONE;
}

aclError aclrtGetDevice(int32_t *deviceId) {
  return ACL_ERROR_NONE;
}

aclError aclrtResetDevice(int32_t deviceId) {
  return ACL_ERROR_NONE;
}

aclError aclrtGetCurrentContext(aclrtContext *context) {
  return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext context) {
  return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif