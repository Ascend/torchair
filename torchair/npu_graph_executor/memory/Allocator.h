#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_MEMORY_BLOCK_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_MEMORY_BLOCK_H_

#include <cstddef>
#include <functional>
#include <mutex>
#include <set>
#include "external/graph/types.h"
#include "ge/ge_allocator.h"
#include "torch_npu/inc/core/NPUBlockHandle.h"
#include "util/object_allocator.h"
#include "tng_status.h"
#include "acl/acl_base.h"

namespace tng {
namespace {
constexpr size_t kMemBlockPoolSize = 10240U;
}
class NpuAllocator;
using DelMemBlockFunc = std::function<void(ge::MemBlock*)>;

class NpuMemBlock : public ge::MemBlock {
 public:
  NpuMemBlock(ge::Allocator &allocator, void *addr, size_t block_size, void *handle)
      : ge::MemBlock(allocator, addr, block_size), handle_(handle) {}
  virtual ~NpuMemBlock() = default;

 private:
  friend class NpuAllocator;
  void *handle_;
};

class NpuAllocator : public ge::Allocator {
 public:
  explicit NpuAllocator(void *stream, size_t capacity = kMemBlockPoolSize)
      : mem_block_pool_(capacity), stream_(stream) {}
  virtual ~NpuAllocator() = default;

  NpuAllocator(const NpuAllocator &) = delete;
  NpuAllocator &operator=(const NpuAllocator &) = delete;
  NpuAllocator(NpuAllocator &&other) = delete;
  NpuAllocator &operator=(NpuAllocator &&other) = delete;

  ge::MemBlock *Malloc(size_t size) override;
  void Free(ge::MemBlock *block) override;
  ge::MemBlock *MallocFeatureMemory(size_t size, const bool is_fixed);
  Status FreeFeatureMemory(ge::MemBlock *block, const bool is_fixed);

 private:
  std::set<ge::MemBlock*> feature_map_mem_pool_;
  std::set<ge::MemBlock*> fixed_mem_pool_;

  ObjectAllocator<NpuMemBlock> mem_block_pool_;
  void *stream_;
  std::mutex allocator_mutex_;
};

}  // namespace tng
#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_MEMORY_BLOCK_H_
