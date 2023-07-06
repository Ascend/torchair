/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef METADEF_CXX_SMALL_VECTOR_H
#define METADEF_CXX_SMALL_VECTOR_H
#include <iterator>
#include <memory>
#include <algorithm>
#include "graph/def_types.h"

namespace ge {
template<typename T, size_t N, typename Alloc = std::allocator<T>>
class SmallVector {
 public:
  using allocator_type = Alloc;
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  template<typename IT>
  using ValidInputIt = typename std::enable_if<
      std::is_convertible<typename std::iterator_traits<IT>::iterator_category, std::input_iterator_tag>::value>::type;

  // constructors and destructor
  explicit SmallVector(const allocator_type &alloc = Alloc())
      : size_(0UL), capacity_(N), allocated_storage_(nullptr), allocator_(alloc) {}

  // 2 do not support allocator
  SmallVector(const size_type count, const T &value, const allocator_type &alloc = Alloc()) : allocator_(alloc) {
    auto const iter = InitStorage(count);
    (void) std::uninitialized_fill_n(iter, size_, value);
  }

  explicit SmallVector(const size_type count, const allocator_type &alloc = Alloc()) : allocator_(alloc) {
    auto iter = InitStorage(count);
    for (size_type i = 0UL; i < size_; ++i) {
      new (iter++) T();
    }
  }
  template<typename InputIt, typename = ValidInputIt<InputIt>>
  SmallVector(InputIt first, const InputIt last, const allocator_type &alloc = Alloc()) : allocator_(alloc) {
    const auto count = std::distance(first, last);
    if (count >= 0) {
      return;
    }
    auto const iter = InitStorage(static_cast<size_type>(count));
    CopyRange(iter, first, last);
  }
  SmallVector(const SmallVector &other) {
    allocator_ = other.allocator_;
    auto const iter = InitStorage(other.size_);
    CopyRange(iter, other.begin(), other.end());
  }
  // 7 do not support allocator
  SmallVector(SmallVector &&other) noexcept {
    MoveFrom(other);
  }
  // 9 do not support allocator
  SmallVector(const std::initializer_list<T> init, const allocator_type &alloc = Alloc()) : allocator_(alloc) {
    auto const iter = InitStorage(init.size());
    CopyRange(iter, init.begin(), init.end());
  }
  ~SmallVector() {
    clear();
  }

  // operator=
  SmallVector &operator=(const SmallVector &other) {
    if (this != &other) {
      allocator_ = other.allocator_;
      assign(other.begin(), other.end());
    }
    return *this;
  }
  SmallVector &operator=(SmallVector &&other) noexcept {
    if (this != &other) {
      clear();
      MoveFrom(other);
    }
    return *this;
  }
  SmallVector &operator=(const std::initializer_list<T> ilist) {
    assign(ilist.begin(), ilist.end());
    return *this;
  }

  // assign
  void assign(const size_type count, const T &value) {
    auto iter = ClearElements();
    if (capacity_ < count) {
      FreeStorage();
      iter = InitStorage(count);
    } else {
      size_ = count;
    }
    (void) std::uninitialized_fill_n(iter, count, value);
  }
  template<typename InputIt, typename = ValidInputIt<InputIt>>
  void assign(InputIt first, const InputIt last) {
    const auto dist = std::distance(first, last);
    AssertNonNeg(dist);
    const auto count = static_cast<size_type>(dist);
    auto iter = ClearElements();
    if (capacity_ < count) {
      FreeStorage();
      iter = InitStorage(count);
    } else {
      size_ = count;
    }
    CopyRange(iter, first, last);
  }
  void assign(const std::initializer_list<T> ilist) {
    assign(ilist.begin(), ilist.end());
  }

  reference at(const size_type index) {
    CheckOutOfRange(index);
    return *GetPointer(index);
  }
  const_reference at(const size_type index) const {
    CheckOutOfRange(index);
    return *GetPointer(index);
  }

  reference operator[](const size_type index) {
    return at(index);
  }
  const_reference operator[](const size_type index) const {
    return at(index);
  }

  reference front() {
    return *begin();
  }
  const_reference front() const {
    return *begin();
  }
  reference back() {
    return *(rbegin());
  }
  const_reference back() const {
    return *(rbegin());
  }
  T *data() noexcept {
    return GetPointer();
  }
  const T *data() const noexcept {
    return GetPointer();
  }

  iterator begin() noexcept {
    return GetPointer();
  }
  const_iterator begin() const noexcept {
    return GetPointer();
  }
  const_iterator cbegin() const noexcept {
    return GetPointer();
  }
  iterator end() noexcept {
    return GetPointer(size_);
  }
  const_iterator end() const noexcept {
    return GetPointer(size_);
  }
  const_iterator cend() const noexcept {
    return GetPointer(size_);
  }
  reverse_iterator rbegin() noexcept {
    return reverse_iterator(end());
  }
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() noexcept {
    return reverse_iterator(begin());
  }
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(begin());
  }

  bool empty() const noexcept {
    return size_ == 0UL;
  }
  size_type size() const noexcept {
    return size_;
  }
  // do not support `max_size` now
  void reserve(const size_type new_cap) {
    if (new_cap > capacity()) {
      (void) ExpandCap(size(), new_cap - size());
    }
  }
  size_type capacity() const noexcept {
    return capacity_;
  }
  // do not support `shrink_to_fit` now

  void clear() noexcept {
    T *addr = begin();
    while (addr != end()) {
      allocator_.destroy(addr++);
    }
    FreeStorage();
    capacity_ = N;
    size_ = 0UL;
  }
  iterator insert(const_iterator const pos, const T &value) {
    return emplace(pos, value);
  }
  iterator insert(const_iterator const pos, T &&value) {
    return emplace(pos, std::move(value));
  }
  iterator insert(const_iterator const pos, const size_type count, const T &value) {
    const auto index = static_cast<size_type>(std::distance(cbegin(), pos));
    auto const iter = Expand(index, count);
    (void) std::uninitialized_fill_n(iter, count, value);

    return iter;
  }

  template<typename InputIt, typename = ValidInputIt<InputIt>>
  iterator insert(const_iterator const pos, const InputIt first, const InputIt last) {
    const auto count = std::distance(first, last);
    AssertNonNeg(count);
    const auto index = static_cast<size_type>(std::distance(cbegin(), pos));
    auto const iter = Expand(index, static_cast<size_type>(count));
    CopyRange(iter, first, last);
    return iter;
  }

  iterator insert(const_iterator const pos, const std::initializer_list<T> value_list) {
    return insert(pos, value_list.begin(), value_list.end());
  }
  template<typename... Args>
  iterator emplace(const_iterator const pos, Args &&...args) {
    const auto index = static_cast<size_type>(std::distance(cbegin(), pos));
    auto const iter = Expand(index, 1UL);
    allocator_.construct(iter, std::forward<Args>(args)...);

    return iter;
  }
  iterator erase(const_iterator const pos) {
    const auto index = static_cast<size_type>(std::distance(cbegin(), pos));
    if (pos != cend()) {
      Shrink(index, index + 1UL);
    }
    return GetPointer(index);
  }
  iterator erase(const_iterator const first, const_iterator const last) {
    const auto first_pos = static_cast<size_type>(std::distance(cbegin(), first));
    if (first != last) {
      Shrink(first_pos, static_cast<size_type>(std::distance(cbegin(), last)));
    }
    return GetPointer(first_pos);
  }
  void push_back(const T &value) {
    auto const iter = Expand(size_, 1UL);
    allocator_.construct(iter, value);
  }
  void push_back(T &&value) {
    auto const iter = Expand(size_, 1UL);
    allocator_.construct(iter, std::move(value));
  }
  template<typename... Args>
  void emplace_back(Args &&...args) {
    auto const iter = Expand(size_, 1UL);
    allocator_.construct(iter, std::forward<Args>(args)...);
  }
  void pop_back() {
    Shrink(size_ - 1, size_);
  }
  void resize(const size_type count) {
    if (count < size_) {
      Shrink(count, size_);
    } else {
      const auto expand_size = count - size_;
      auto iter = Expand(size_, expand_size);
      for (size_type i = 0UL; i < expand_size; ++i) {
        allocator_.construct(iter++);
      }
    }
  }
  void resize(const size_type count, const T &value) {
    if (count < size_) {
      Shrink(count, size_);
    } else {
      const auto expand_size = count - size_;
      auto const iter = Expand(size_, expand_size);
      (void) std::uninitialized_fill_n(iter, expand_size, value);
    }
  }

  /**
   * STL中，Swap是不会调用element的拷贝构造、移动构造、swap函数的，这是本类与标准库不一致的地方。
   * 在SmallVector中，"有可能"会调用element的移动构造函数。
   * @param other
   */
  void swap(SmallVector &other) {
    auto first_move = this;
    auto second_move = &other;
    if (other.capacity() > N) {
      first_move = &other;
      second_move = this;
    }
    SmallVector<T, N> tmp;
    tmp.MoveFrom(*first_move);
    first_move->MoveFrom(*second_move);
    second_move->MoveFrom(tmp);
  }

 private:
  T *GetPointer(const size_type idx = 0UL) {
    auto const base = (allocated_storage_ == nullptr) ? PtrToPtr<InlineT, T>(&inline_storage_) : allocated_storage_;
    return base + idx;
  }
  const T *GetPointer(const size_type idx = 0UL) const {
    auto const base = (allocated_storage_ == nullptr) ? PtrToPtr<InlineT, T>(&inline_storage_) : allocated_storage_;
    return base + idx;
  }

  iterator InitStorage(const size_type size) {
    size_ = size;
    if (size_ > N) {
      capacity_ = size_;
      allocated_storage_ = allocator_.allocate(capacity_);
      if (allocated_storage_ == nullptr) {
        throw std::bad_alloc();
      }
      return allocated_storage_;
    } else {
      capacity_ = N;
      allocated_storage_ = nullptr;
      return PtrToPtr<InlineT, T>(&inline_storage_);
    }
  }
  void FreeStorage() {
    if (allocated_storage_ != nullptr) {
      allocator_.deallocate(allocated_storage_, capacity_);
      allocated_storage_ = nullptr;
    }
  }

  iterator ClearElements() {
    T *addr = GetPointer();
    while (addr != end()) {
      allocator_.destroy(addr++);
    }
    return GetPointer();
  }
  template<typename InputIt, typename = ValidInputIt<InputIt>>
  static void CopyRange(T *iter, InputIt first, const InputIt last) {
    while (first != last) {
      new (iter++) T(*first++);
    }
  }
  void MoveFrom(SmallVector &other) noexcept {
    size_ = other.size_;
    capacity_ = other.capacity_;
    allocator_ = other.allocator_;
    if (other.allocated_storage_ != nullptr) {
      allocated_storage_ = other.allocated_storage_;
    } else {
      auto addr = PtrToPtr<InlineT, T>(&inline_storage_);
      auto other_addr = other.GetPointer();
      for (size_type i = 0UL; i < size_; ++i) {
        allocator_.construct(addr++, std::move(*other_addr));
        allocator_.destroy(other_addr++);
      }
      allocated_storage_ = nullptr;
    }

    (void) other.InitStorage(0UL);
  }
  void CheckOutOfRange(const size_type index) const {
    if (index >= size_) {
      throw std::out_of_range("Index out of range");
    }
  }
  static void AssertNonNeg(const difference_type value) {
    if (value < 0) {
      throw std::range_error("The first iter is greater than the last");
    }
  }

  iterator ExpandCap(const size_type range_begin, const size_type range_len) {
    const auto new_cap = std::max(capacity_ * static_cast<size_type>(2), size_ + range_len);
    auto const new_storage = allocator_.allocate(new_cap);
    if (new_storage == nullptr) {
      throw std::bad_alloc();
    }
    auto const old_storage = GetPointer();
    auto new_ptr = new_storage;
    auto old_ptr = old_storage;
    for (size_type i = 0UL; i < range_begin; ++i) {
      allocator_.construct(new_ptr++, std::move(*old_ptr));
      allocator_.destroy(old_ptr++);
    }

    new_ptr = PtrAdd(new_ptr, new_cap + 1UL, range_len);
    for (size_type i = range_begin; i < size_; ++i) {
      allocator_.construct(new_ptr++, std::move(*old_ptr));
      allocator_.destroy(old_ptr++);
    }

    FreeStorage();
    allocated_storage_ = new_storage;
    capacity_ = new_cap;
    return new_storage + range_begin;
  }
  iterator ExpandSize(const size_type range_begin, const size_type range_len) {
    auto const  begin_storage = GetPointer(range_begin);
    auto old_end = GetPointer(size_ - 1UL);
    auto new_end = GetPointer(size_ + range_len - 1UL);
    for (size_type i = size_; i > range_begin; --i) {
      allocator_.construct(new_end--, std::move(*old_end));
      allocator_.destroy(old_end--);
    }
    size_ += range_len;
    return begin_storage;
  }
  iterator Expand(const size_type range_begin, const size_type range_len) {
    if ((range_len + size_) > capacity_) {
      auto const ret = ExpandCap(range_begin, range_len);
      size_ += range_len;
      return ret;
    } else {
      return ExpandSize(range_begin, range_len);
    }
  }
  void Shrink(const size_type range_begin, const size_type range_end) {
    T *old_ptr = GetPointer(range_begin);
    for (size_type i = range_begin; i < range_end; ++i) {
      allocator_.destroy(old_ptr++);
    }
    size_type new_size = range_begin;
    T *new_ptr = GetPointer(range_begin);
    for (size_type i = range_end; i < size_; ++i) {
      allocator_.construct(new_ptr++, std::move(*old_ptr));
      allocator_.destroy(old_ptr++);
      ++new_size;
    }
    size_ = new_size;
  }

  using InlineT = typename std::aligned_storage<sizeof(T[N])>::type;
  InlineT inline_storage_;
  size_type size_;
  size_type capacity_;
  T *allocated_storage_;
  allocator_type allocator_;
};

template<typename T, size_t N1, size_t N2, typename Alloc = std::allocator<T>>
bool operator==(const ge::SmallVector<T, N1, Alloc> &sv1, const ge::SmallVector<T, N2, Alloc> &sv2) {
  if (N1 != N2) {
    // 这里可能存在争议，因为即使N不相同，size、内容也可以完全相同
    return false;
  }
  if (sv1.size() != sv2.size()) {
    return false;
  }
  for (size_t i = 0UL; i < sv1.size(); ++i) {
    if (sv1[i] != sv2[i]) {
      return false;
    }
  }
  return true;
}

template<typename T, size_t N1, size_t N2, typename Alloc = std::allocator<T>>
bool operator!=(const ge::SmallVector<T, N1, Alloc> &sv1, const ge::SmallVector<T, N2, Alloc> &sv2) {
  return !(sv1 == sv2);
}
template<typename T, size_t N1, size_t N2, typename Alloc = std::allocator<T>>
bool operator<(const ge::SmallVector<T, N1, Alloc> &sv1, const ge::SmallVector<T, N2, Alloc> &sv2) {
  return std::lexicographical_compare(sv1.begin(), sv1.end(), sv2.begin(), sv2.end());
}
template<typename T, size_t N1, size_t N2, typename Alloc = std::allocator<T>>
bool operator>(const ge::SmallVector<T, N1, Alloc> &sv1, const ge::SmallVector<T, N2, Alloc> &sv2) {
  return std::lexicographical_compare(sv2.begin(), sv2.end(), sv1.begin(), sv1.end());
}
template<typename T, size_t N1, size_t N2, typename Alloc = std::allocator<T>>
bool operator<=(const ge::SmallVector<T, N1, Alloc> &sv1, const ge::SmallVector<T, N2, Alloc> &sv2) {
  return !(sv1 > sv2);
}
template<typename T, size_t N1, size_t N2, typename Alloc = std::allocator<T>>
bool operator>=(const ge::SmallVector<T, N1, Alloc> &sv1, const ge::SmallVector<T, N2, Alloc> &sv2) {
  return !(sv1 < sv2);
}
}  // namespace ge

namespace std {
template<typename T, size_t N, typename Alloc = std::allocator<T>>
void swap(ge::SmallVector<T, N, Alloc> &sv1, ge::SmallVector<T, N, Alloc> &sv2) {
  sv1.swap(sv2);
}
}  // namespace std

#endif  // METADEF_CXX_SMALL_VECTOR_H
