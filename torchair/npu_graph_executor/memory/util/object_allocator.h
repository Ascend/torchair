#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_MEMORY_UTIL_OBJECT_ALLOCATOR_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_MEMORY_UTIL_OBJECT_ALLOCATOR_H_
#include <new>
#include "link.h"
#include "link_node.h"
#include "checker.h"

namespace tng {
template <typename T>
class ObjectAllocator {
 public:
  explicit ObjectAllocator(size_t capacity) {
    for (size_t i = 0U; i < capacity; i++) {
      auto elem = new (std::nothrow) Element();
      if (elem != nullptr) {
        elems.push_back(elem->node);
      }
    }
  }

  virtual ~ObjectAllocator() {
    while (!elems.empty()) {
      auto elem = elems.pop_front();
      if (elem != nullptr) {
        delete reinterpret_cast<Element *>(elem);
      }
    }
  }

  // Alloc memory but do not construct !!!
  T *AllocMem() {
    Element *elem = reinterpret_cast<Element *>(elems.pop_front());
    if (!elem) {
      elem = new (std::nothrow) Element();
    }
    TNG_ASSERT_NOTNULL(elem, "elem is nullptr.");
    return reinterpret_cast<T *>(elem->buff);
  }

  // Free memory but do not destruct !!!
  void FreeMem(T &elem) {
    elems.push_front(*(reinterpret_cast<ElemNode *>(&elem)));
  }

  // Alloc memory and construct with args!!!
  template <class... Args>
  T *New(Args &&...args) {
    return new (AllocMem()) T(std::forward<Args>(args)...);
  }

  // Alloc memory and construct without args!!!
  T *Alloc() {
    auto elem = elems.pop_front();
    if (elem != nullptr) {
      return reinterpret_cast<T *>(elem);
    }
    return reinterpret_cast<T *>(new (std::nothrow) Element());
  }

  // Free memory and destruct !!!
  void Free(T &elem) {
    elem.~T();
    FreeMem(elem);
  }

  size_t GetAvailableSize() const {
    return elems.size();
  }

 private:
  struct ElemNode : LinkNode<ElemNode> {};

  union Element {
    Element() {}
    ElemNode node;
    uint8_t buff[sizeof(T)];
  };

 private:
  Link<ElemNode> elems;
};
}  // namespace tng
#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_MEMORY_UTIL_OBJECT_ALLOCATOR_H_
