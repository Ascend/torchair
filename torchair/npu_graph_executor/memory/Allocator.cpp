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
  TNG_LOG(DEBUG) << "[MemoryTrace] Malloc memory from NPUCachingAllocator success, block = " << block;
  ge::MemBlock *mem_block = nullptr;
  {
    const std::unique_lock<std::mutex> lk(allocator_mutex_);
    mem_block = new (mem_block_pool_.Alloc()) NpuMemBlock(*this, GetBlockPtr(block), GetBlockSize(block), block);
  }
  if (mem_block == nullptr) {
    TNG_LOG(ERROR) << "Failed to create ge_block from memory block pool";
    return nullptr;
  }
  TNG_LOG(DEBUG) << "[MemoryTrace] Malloc the mem_block success, mem_block = " << mem_block
                << ", device_ptr = " << mem_block->GetAddr() << ", size = " << mem_block->GetSize();
  return mem_block;
}

void NpuAllocator::Free(ge::MemBlock *block) {
  auto mem_block = dynamic_cast<NpuMemBlock *>(block);
  if (mem_block == nullptr) {
    if (block == nullptr) {
      TNG_LOG(DEBUG) << "Try to free nullptr block failed, due to memory block is nullptr, too";
    } else {
      TNG_LOG(WARNING) << "Try to free block" << block
                       << " failed, due to memory block is not belong to mem_block_pool_";
    }
    return;
  }
  // free mem_block
  TNG_LOG(DEBUG) << "[MemoryTrace] Try to free the mem_block, mem_block = " << mem_block
                << ", NPUCachingAllocator block = " << mem_block->handle_
                << ", device_ptr = " << GetBlockPtr(mem_block->handle_)
                << ", size = " << GetBlockSize(mem_block->handle_);
  c10_npu::NPUCachingAllocator::FreeBlock(mem_block->handle_);
  {
    const std::unique_lock<std::mutex> lk(allocator_mutex_);
    mem_block_pool_.Free(*(mem_block));
  }
  TNG_LOG(DEBUG) << "[MemoryTrace]Free the mem_block success.";
}

ge::MemBlock *NpuAllocator::MallocFeatureMemory(size_t size, const bool is_fixed) {
  auto MallocPoolMemory = [this](size_t size, std::set<ge::MemBlock*> &mem_pool) -> ge::MemBlock* {
    for (auto &iter : mem_pool) {
      if (iter->GetSize() >= size) {
        iter->AddCount();
        TNG_LOG(INFO) << "[MemoryTrace] MallocPoolMemory: Reuse memory "
                      << ", block = " << iter << " , addr = " << iter->GetAddr()
                      << ", and size = " << iter->GetSize() << " , use count = " << iter->GetCount();
        return iter;
      }
    }
    TNG_LOG(INFO) << "[MemoryTrace] MallocPoolMemory: Try Malloc size = " << size;
    ge::MemBlock *block = Malloc(size);
    if (block == nullptr) {
      TNG_LOG(ERROR) << "[MemoryTrace] MallocPoolMemory to malloc memory, size: " << size;
      return nullptr;
    }
    TNG_LOG(INFO) << "[MemoryTrace] MallocPoolMemory: Malloc memory success, size = " << block->GetSize()
                  << ", and addr = " << block->GetAddr() << ", use count = " << block->GetCount();
    return *(mem_pool.insert(block).first);
  };

  if (is_fixed) {
    TNG_LOG(INFO) << "[MemoryTrace] MallocFixedMemory: Try Malloc size = " << size;
    return MallocPoolMemory(size, fixed_mem_pool_);
  } else {
    TNG_LOG(INFO) << "[MemoryTrace] MallocFeatureMemory: Try Malloc size = " << size;
    return MallocPoolMemory(size, feature_map_mem_pool_);
  }
}

Status NpuAllocator::FreeFeatureMemory(ge::MemBlock *block, const bool is_fixed) {
  TNG_ASSERT_NOTNULL(block);
  auto FreePoolMemory = [this](ge::MemBlock *block, std::set<ge::MemBlock*> &mem_pool) -> Status {
    TNG_LOG(INFO) << "[MemoryTrace] FreePoolMemory: Try to Free memory, size = " << block->GetSize()
                  << ", and addr = " << block->GetAddr() << " , use count = " << block->GetCount();
    const auto &iter = mem_pool.find(block);
    if (iter == mem_pool.end()) {
      TNG_LOG(INFO) << "Can not find block = " << block << " from memory pool in current allocator";
      return Status::Success();
    }
    block->Free();
    if (block->GetCount() == 0U) {
      mem_pool.erase(iter);
    }
    TNG_LOG(INFO) << "[MemoryTrace] FreePoolMemory: Free memory success, block use count = "
                  << block->GetCount() << " , memory pool size = " << mem_pool.size();
    block = nullptr;
    return Status::Success();
  };

  if (is_fixed) {
    TNG_LOG(INFO) << "[MemoryTrace] FreeFixedMemory: Try to Free memory, size = " << block->GetSize()
                  << ", and addr = " << block->GetAddr() << " , use count = " << block->GetCount();
    TNG_RETURN_IF_ERROR(FreePoolMemory(block, fixed_mem_pool_));
  } else {
    TNG_LOG(INFO) << "[MemoryTrace] FreeFeatureMemory: Try to Free memory, size = " << block->GetSize()
                  << ", and addr = " << block->GetAddr() << " , use count = " << block->GetCount();
    TNG_RETURN_IF_ERROR(FreePoolMemory(block, feature_map_mem_pool_));
  }
  return Status::Success();
}

}  // namespace tng
