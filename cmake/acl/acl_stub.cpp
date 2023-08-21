#include <iostream>
#include "acl/acl_rt.h"

aclError aclrtSetDevice(int32_t deviceId) {
  std::cerr << "[STUB] aclrtSetDevice" << std::endl;
  return ACL_ERROR_NONE;
}

aclError aclrtGetDevice(int32_t *deviceId) {
  std::cerr << "[STUB] aclrtGetDevice" << std::endl;
  return ACL_ERROR_NONE;
}

aclError aclrtResetDevice(int32_t deviceId) {
  std::cerr << "[STUB] aclrtResetDevice" << std::endl;
  return ACL_ERROR_NONE;
}