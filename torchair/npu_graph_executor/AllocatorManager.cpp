#include "AllocatorManager.h"
#include "checker.h"
#include "ge/ge_allocator.h"
#include "logger.h"
#include "memory/Allocator.h"
#include "session.h"
#include "torch_npu/inc/core/NPUStream.h"

namespace tng {
Status AllocatorManager::EnsureAllocatorRegistered(void *stream) {
  TNG_LOG(INFO) << "Start to EnsureAllocatorRegistered according to stream = " << stream;
  const std::unique_lock<std::mutex> lk(allocators_mutex_);
  auto iter = stream_allocator_registered_.find(stream);
  if (iter != stream_allocator_registered_.end()) {
    TNG_LOG(INFO) << "External allocator has registered, stream = " << iter->first << " , allocator = " << iter->second;
    return Status::Success();
  }
  std::shared_ptr<NpuAllocator> allocator_ptr = std::make_shared<NpuAllocator>(stream);
  TNG_ASSERT(allocator_ptr);
  TNG_LOG(INFO) << "External allocator did not registered, register allocator = " << allocator_ptr
                << " while stream = " << stream;
  stream_allocator_registered_[stream] = allocator_ptr;
  return Session::GetInstance().RegisterExternalAllocator(stream, allocator_ptr);
}

}  // namespace tng
