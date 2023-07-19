#include "Allocator.h"
#include <memory>
#include "checker.h"
#include "logger.h"
#include "torch_npu/inc/core/NPUStream.h"

namespace tng {
using namespace c10_npu::NPUCachingAllocator;

ge::MemBlock *NpuAllocator::Malloc(size_t size) {
  void *block = c10_npu::NPUCachingAllocator::MallocBlock(size, stream_);
  if (block == nullptr) {
    TNG_LOG(ERROR) << "Failed to malloc memory by allocator, size: " << size;
    return nullptr;
  }
  TNG_LOG(INFO) << "[MemoryTrace]Malloc memory from NPUCachingAllocator success, block = " << block;
  ge::MemBlock *mem_block = nullptr;
  {
    const std::unique_lock<std::mutex> lk(allocator_mutex_);
    mem_block = new (mem_block_pool_.Alloc()) NpuMemBlock(*this, GetBlockPtr(block), GetBlockSize(block), block);
  }
  if (mem_block == nullptr) {
    TNG_LOG(ERROR) << "Failed to create ge_block from memory block pool";
    return nullptr;
  }
  TNG_LOG(INFO) << "[MemoryTrace]Malloc the mem_block success, mem_block = " << mem_block
                << ", device_ptr = " << mem_block->GetAddr() << ", size = " << mem_block->GetSize();
  return mem_block;
}

void NpuAllocator::Free(ge::MemBlock *block) {
  auto mem_block = dynamic_cast<NpuMemBlock *>(block);
  if (mem_block == nullptr) {
    if (block == nullptr) {
      TNG_LOG(INFO) << "Try to free nullptr block failed, due to memory block is nullptr, too";
    } else {
      TNG_LOG(WARNING) << "Try to free block" << block
                       << " failed, due to memory block is not belong to mem_block_pool_";
    }
    return;
  }
  // free mem_block
  TNG_LOG(INFO) << "[MemoryTrace]Try to free the mem_block, mem_block = " << mem_block
                << ", device_ptr = " << GetBlockPtr(mem_block->handle_)
                << ", size = " << GetBlockSize(mem_block->handle_);
  c10_npu::NPUCachingAllocator::FreeBlock(mem_block->handle_);
  {
    const std::unique_lock<std::mutex> lk(allocator_mutex_);
    mem_block_pool_.Free(*(mem_block));
  }
  TNG_LOG(INFO) << "[MemoryTrace]Free the mem_block success!";
}
}  // namespace tng
