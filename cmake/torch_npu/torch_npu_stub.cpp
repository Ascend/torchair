#include <iostream>
#include <unordered_map>
#include "core/NPUBlockHandle.h"
#include "core/NPUStream.h"

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
  }
  return const_cast<void*>(iter->first);
}

size_t GetBlockSize(const void *handle) {
  auto iter = global_mem_pool_.find(handle);
  if (iter == global_mem_pool_.end()) {
    std::cerr << "[STUB] GetBlockSize failed with addr " << handle << std::endl;
  }
  return iter->second;
}

} // namespace NPUCachingAllocator
}  // namespace c10_npu
