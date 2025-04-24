#include <iostream>
#include <unordered_map>
#include "torch/torch.h"
#include "core/NPUBlockHandle.h"
#include "core/NPUFormat.h"
#include "core/NPUStream.h"
#include "core/GetCANNInfo.h"

namespace c10_npu {
namespace {
c10::Device device(c10::DeviceType::PrivateUse1, 0);
c10::Stream default_stream = c10::Stream(c10::Stream::Default::DEFAULT, device);
}  // namespace

NPUStream getCurrentNPUStream(c10::DeviceIndex device_index) { return NPUStream(default_stream); }

aclrtStream NPUStream::stream(const bool need_empty) const {
    return reinterpret_cast<void *>(0x1);
}

std::ostream &operator<<(std::ostream &stream, const NPUStream &s) { return stream << s.unwrap(); }

namespace NPUCachingAllocator {

std::unordered_map<const void*, size_t> global_mem_pool_;
void *MallocBlock(size_t size, void *stream, int device) {
  std::cerr << "[STUB] try to mallocBlock addr with size = " << size << std::endl;
  (void)stream;
  (void)device;
  auto addr = new (std::nothrow) uint8_t[size]();
  if (addr == nullptr) {
    std::cerr << "[STUB] Malloc addr failed with size " << size << std::endl;
    return nullptr;
  }
  global_mem_pool_.insert(std::make_pair(addr, size));
  return reinterpret_cast<void*>(addr);
}

void FreeBlock(void *handle) {
  std::cerr << "[STUB] try to free block addr with handle = " << handle << std::endl;
  auto iter = global_mem_pool_.find(handle);
  if (iter == global_mem_pool_.end()) {
    std::cerr << "[STUB] FreeBlock failed with addr " << handle << std::endl;
    return;
  }
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>((iter->first));
  delete[] ptr;
  ptr = nullptr;
  global_mem_pool_.erase(iter);
  return;
}

void *GetBlockPtr(const void *handle) {
  auto iter = global_mem_pool_.find(handle);
  if (iter == global_mem_pool_.end()) {
    std::cerr << "[STUB] GetBlockPtr failed with addr " << handle << std::endl;
    return nullptr;
  }
  return const_cast<void*>(iter->first);
}

size_t GetBlockSize(const void *handle) {
  auto iter = global_mem_pool_.find(handle);
  if (iter == global_mem_pool_.end()) {
    std::cerr << "[STUB] GetBlockSize failed with addr " << handle << std::endl;
    return 0U;
  }
  return iter->second;
}

} // namespace NPUCachingAllocator
}  // namespace c10_npu

namespace at_npu {
namespace native {

int64_t get_npu_format(const at::Tensor &tensor) {
  // ACL_FORMAT_ND = 2
  return 2;
}

std::vector<int64_t> get_npu_storage_sizes(const at::Tensor &tensor) {
  auto sizes = tensor.sizes();
  std::vector<int64_t> vector_storage_sizes(sizes.begin(), sizes.end());
  return vector_storage_sizes;
}

at::Tensor npu_format_cast(const at::Tensor &tensor, int64_t acl_format) {
  return tensor.clone();
}

at::Tensor empty_with_format(c10::IntArrayRef sizes, const c10::TensorOptions &options, int64_t format,
                             bool keep_format) {
  return at::empty(sizes, options);
}

}  // namespace native
}  // namespace at_npu

std::string GetCANNVersion(const std::string& module) {
  return "8.1.RC1.alpha1";
}
