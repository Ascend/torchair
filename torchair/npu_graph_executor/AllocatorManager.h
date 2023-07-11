#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_ALLOCATOR_MANAGER_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_ALLOCATOR_MANAGER_H_

#include <memory>
#include <mutex>
#include <unordered_map>
#include "ge/ge_allocator.h"
#include "tng_status.h"
#include "external/graph/types.h"

namespace tng {
class AllocatorManager {
 public:
  static AllocatorManager &GetInstance() {
    static AllocatorManager instance;
    return instance;
  }

  AllocatorManager(const AllocatorManager &) = delete;
  AllocatorManager &operator=(const AllocatorManager &) = delete;
  AllocatorManager(AllocatorManager &&other) = delete;
  AllocatorManager &operator=(AllocatorManager &&other) = delete;

  Status EnsureAllocatorRegistered(void *stream);

 private:
  AllocatorManager() = default;
  ~AllocatorManager() = default;
  std::mutex allocators_mutex_;
  std::unordered_map<void *, std::shared_ptr<ge::Allocator>> stream_allocator_registered_;
};
}  // namespace tng
#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_ALLOCATOR_MANAGER_H_
