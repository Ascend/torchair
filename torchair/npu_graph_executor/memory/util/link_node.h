#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_MEMORY_UTIL_LINK_NODE_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_MEMORY_UTIL_LINK_NODE_H_

namespace tng {
template <typename T>
struct List;

template <typename T>
struct LinkNode {
  LinkNode() {
    link_.prev_ = nullptr;
    link_.next_ = nullptr;
  }

  void remove() {
    // Notice: Just used in scenes careless num of link!!!
    link_.prev_->link_.next_ = link_.next_;
    link_.next_->link_.prev_ = link_.prev_;
  }

  T *next() {
    return link_.next_;
  }

  const T *next() const {
    return link_.next_;
  }

  T *prev() {
    return link_.prev_;
  }

  const T *prev() const {
    return link_.prev_;
  }

  friend struct List<T>;

  struct Chain {
    T *volatile next_;
    T *volatile prev_;
  };  // __cacheline_aligned;

  Chain link_;
};
}  // namespace tng
#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_MEMORY_UTIL_LINK_NODE_H_
