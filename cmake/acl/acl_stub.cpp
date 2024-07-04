#include "acl/acl_rt.h"
#include "acl/acl_op_compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t COMPILE_OPT_SIZE = 256U;

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

aclError aclrtMemcpy(void *dst, size_t destMax,
                     const void *src,
                     size_t count,
                     aclrtMemcpyKind kind) {
    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsync(void *dst, size_t destMax,
                          const void *src,
                          size_t count,
                          aclrtMemcpyKind kind,
                          aclrtStream stream) {
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeDevice(void) {
  return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream stream) {
    return ACL_SUCCESS;
}

size_t aclGetCompileoptSize(aclCompileOpt opt) {
  return COMPILE_OPT_SIZE;
}

aclError aclGetCompileopt(aclCompileOpt opt, char *value, size_t length) {
  return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif