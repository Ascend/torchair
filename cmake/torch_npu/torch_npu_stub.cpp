#include <iostream>
#include "core/NPUStream.h"
#include "core/NPUBlockHandle.h"

namespace c10_npu {
namespace {
c10::Device device(c10::kCPU, 0);
c10::Stream default_stream = c10::Stream(c10::Stream::Default::DEFAULT, device);
}  // namespace

NPUStream getCurrentNPUStream(c10::DeviceIndex device_index) { return NPUStream(default_stream); }

std::ostream &operator<<(std::ostream &stream, const NPUStream &s) { return stream << s.unwrap(); }

namespace NPUCachingAllocator {
void *MallocBlock(size_t size, void *stream, int device) { return nullptr; }

void FreeBlock(void *handle) { return; }

void *GetBlockPtr(const void *handle) { return nullptr; }

size_t GetBlockSize(const void *handle) { return 0U; }

} // namespace NPUCachingAllocator
}  // namespace c10_npu
