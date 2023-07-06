/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#ifndef METADEF_CXX_INC_EXE_GRAPH_RANGE_H_
#define METADEF_CXX_INC_EXE_GRAPH_RANGE_H_

#include <array>
#include <iostream>
#include "utils/extern_math_util.h"
#include "shape.h"

namespace gert {
template<typename T>
class Range {
 public:
  /**
   * 默认构造一个range
   */
  Range() : min_(nullptr), max_(nullptr) {
    memset(reserved_, 0, sizeof(reserved_));
  }

  /**
   * 通过最小和最大T对象指针构造range
   * @param min range最小值指针
   * @param max range最大值指针
   */
  Range(T *min, T* max) : min_(min), max_(max) {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }

  /**
   * 通过一个T指针构造range，表示最大最小值相同
   * @param same_ele T指针
   */
  explicit Range(T *same_ele) : min_(same_ele), max_(same_ele) {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }

  /**
   * 判断与另外一个range对象是否相等，如果两个range的最小和最大元素指针分别相等，那么认为两个range相等，
   * 如果存在指针不相等的情况，再对比T对象是否分别相等
   * @param rht 另一个Range对象
   * @return true/false
   */
  bool operator==(const Range<T> &rht) const {
    if ((this->min_ == rht.min_) && (this->max_ == rht.max_)) {
      return true;
    } else {
      return (*this->min_ == *rht.min_) && (*this->max_ == *rht.max_);
    }
  }

  /**
   * 设置最小的T对象指针
   * @param min 最小的T对象指针
   */
  void SetMin(T *min) {
    min_ = min;
  }

  /**
   * 设置最大的T对象指针
   * @param max 最大的T对象指针
   */
  void SetMax(T *max) {
    max_ = max;
  }

  /**
   * 获取最小的T对象指针
   * @return
   */
  const T *GetMin() const {
    return min_;
  }

  /**
   * 获取最大的T对象指针
   * @return
   */
  const T *GetMax() const {
    return max_;
  }

  /**
   * 获取最小的T对象指针
   * @return
   */
  T *GetMin() {
    return min_;
  }

  /**
   * 获取最大的T对象指针
   * @return
   */
  T *GetMax() {
    return max_;
  }
 private:
  T *min_;
  T *max_;
  uint8_t reserved_[40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_RANGE_H_
