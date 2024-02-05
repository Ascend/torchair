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
  TNG_LOG(INFO) << "[MemoryTrace] Malloc memory from NPUCachingAllocator success, block = " << block;
  ge::MemBlock *mem_block = nullptr;
  {
    const std::unique_lock<std::mutex> lk(allocator_mutex_);
    mem_block = new (mem_block_pool_.Alloc()) NpuMemBlock(*this, GetBlockPtr(block), GetBlockSize(block), block);
  }
  if (mem_block == nullptr) {
    TNG_LOG(ERROR) << "Failed to create ge_block from memory block pool";
    return nullptr;
  }
  TNG_LOG(INFO) << "[MemoryTrace] Malloc the mem_block success, mem_block = " << mem_block
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
  TNG_LOG(INFO) << "[MemoryTrace] Try to free the mem_block, mem_block = " << mem_block
                << ", NPUCachingAllocator block = " << mem_block->handle_
                << ", device_ptr = " << GetBlockPtr(mem_block->handle_)
                << ", size = " << GetBlockSize(mem_block->handle_);
  c10_npu::NPUCachingAllocator::FreeBlock(mem_block->handle_);
  {
    const std::unique_lock<std::mutex> lk(allocator_mutex_);
    mem_block_pool_.Free(*(mem_block));
  }
  TNG_LOG(INFO) << "[MemoryTrace]Free the mem_block success.";
}

ge::MemBlock *NpuAllocator::MallocFeatureMemory(size_t size, ge::MemBlock *advised_block) {
  if (advised_block == nullptr) {  // first run
    for (auto &iter : feature_map_mem_pool_) {
      if (iter->GetSize() >= size) {
        iter->AddCount();
        TNG_LOG(INFO) << "[MemoryTrace] MallocFeatureMemory: Reuse feature memory "
                      << ", block = " << iter << " , addr = " << iter->GetAddr() << ", and size = " << iter->GetSize()
                      << " , use count = " << iter->GetCount();
        return iter;
      }
    }
    TNG_LOG(INFO) << "[MemoryTrace] MallocFeatureMemory: Try Malloc size = " << size << ", advised_block = " << advised_block;
    ge::MemBlock *block = Malloc(size);
    if (block == nullptr) {
      TNG_LOG(ERROR) << "[MemoryTrace]MallocFeatureMemoryFailed to malloc feature memory, size: " << size;
      return nullptr;
    }
    TNG_LOG(INFO) << "[MemoryTrace] MallocAdvise: Malloc memory success, size = " << block->GetSize()
                  << ", and addr = " << block->GetAddr() << ", use count = " << block->GetCount();
    return *(feature_map_mem_pool_.insert(block).first);
  } else {  // After the first run
    auto iter = feature_map_mem_pool_.find(advised_block);
    if (iter == feature_map_mem_pool_.end()) {
      TNG_LOG(ERROR) << "Failed to find feature map memory from feature memory pool, size = " << size
                     << " , block = " << advised_block;
      return nullptr;
    }
    TNG_LOG(INFO) << "[MemoryTrace]MallocAdvise: find feature map memory from feature memory pool success, size = "
                  << size << ", and addr = " << (*iter)->GetAddr();
    return *iter;
  }
}

Status NpuAllocator::FreeFeatureMemory(ge::MemBlock *block) {
  TNG_ASSERT_NOTNULL(block);
  TNG_LOG(INFO) << "[MemoryTrace] FreeFeatureMemory: Try to Free memory, size = " << block->GetSize()
                << ", and addr = " << block->GetAddr() << " , use count = " << block->GetCount();
  auto iter = feature_map_mem_pool_.find(block);
  if (iter == feature_map_mem_pool_.end()) {
    TNG_LOG(INFO) << "Can not find block = " << block << " from feature map memory pool in current allocator";
    return Status::Success();
  }
  block->Free();
  if (block->GetCount() == 0U) {
    feature_map_mem_pool_.erase(iter);
  }
  TNG_LOG(INFO) << "[MemoryTrace] FreeFeatureMemory: Free feature map memory success, block use count = "
                << block->GetCount() << " , feature map memory pool size = " << feature_map_mem_pool_.size();
  block = nullptr;
  return Status::Success();
}
}  // namespace tng
