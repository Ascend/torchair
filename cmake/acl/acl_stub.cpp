#include <thread>
#include <chrono>
#include <string>
#include "acl/acl_rt.h"
#include "acl/acl_tdt.h"
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

acltdtTensorType acltdtGetTensorTypeFromItem(const acltdtDataItem *dataItem) {
  return acltdtTensorType::ACL_TENSOR_DATA_ABNORMAL;
}

aclDataType acltdtGetDataTypeFromItem(const acltdtDataItem *dataItem) {
  return aclDataType::ACL_DT_UNDEFINED;
}

static std::string kMsg = "good job";

void *acltdtGetDataAddrFromItem(const acltdtDataItem *dataItem) {
  return static_cast<void *>(const_cast<char *>(kMsg.c_str()));
}

size_t acltdtGetDataSizeFromItem(const acltdtDataItem *dataItem) {
  return kMsg.size();
}

size_t acltdtGetDimNumFromItem(const acltdtDataItem *dataItem) {
  return 0;
}

aclError acltdtGetSliceInfoFromItem(const acltdtDataItem *dataItem, size_t *sliceNum,
                                    size_t *sliceId) {
  return ACL_ERROR_NONE;
}

aclError acltdtGetDimsFromItem(const acltdtDataItem *dataItem, int64_t *dims, size_t dimNum) {
  return ACL_ERROR_NONE;
}

acltdtDataItem *acltdtCreateDataItem(acltdtTensorType tdtType,
                                     const int64_t *dims,
                                     size_t dimNum,
                                     aclDataType dataType,
                                     void *data,
                                     size_t size) {
  static int64_t handle = 0;
  return reinterpret_cast<acltdtDataItem *>(&handle);
}

aclError acltdtDestroyDataItem(acltdtDataItem *dataItem) {
  return ACL_ERROR_NONE;
}

acltdtDataset *acltdtCreateDataset() {
  static int64_t handle = 0;
  return reinterpret_cast<acltdtDataset *>(&handle);
}

aclError acltdtDestroyDataset(acltdtDataset *dataset) {
  return ACL_ERROR_NONE;
}

acltdtDataItem *acltdtGetDataItem(const acltdtDataset *dataset, size_t index) {
  static int64_t handle = 0;
  return reinterpret_cast<acltdtDataItem *>(&handle);
}

aclError acltdtAddDataItem(acltdtDataset *dataset, acltdtDataItem *dataItem) {
  return ACL_ERROR_NONE;
}

size_t acltdtGetDatasetSize(const acltdtDataset *dataset) {
  return 1;
}

const char *acltdtGetDatasetName(const acltdtDataset *dataset) {
  return kMsg.c_str();
}

aclError acltdtStopChannel(acltdtChannelHandle *handle) {
  return ACL_ERROR_NONE;
}

acltdtChannelHandle *acltdtCreateChannelWithCapacity(uint32_t deviceId,
                                                     const char *name,
                                                     size_t capacity) {
  static int64_t handle = 0;
  return reinterpret_cast<acltdtChannelHandle *>(&handle);
}

aclError acltdtDestroyChannel(acltdtChannelHandle *handle) {
  return ACL_ERROR_NONE;
}

aclError acltdtReceiveTensor(const acltdtChannelHandle *handle,
                             acltdtDataset *dataset,
                             int32_t timeout) {
  std::this_thread::sleep_for(std::chrono::seconds(1));
  return ACL_ERROR_NONE;
}

aclError acltdtQueryChannelSize(const acltdtChannelHandle *handle, size_t *size) {
  return ACL_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif