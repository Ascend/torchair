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
#ifndef METADEF_CXX_INC_EXE_GRAPH_STORAGE_SHAPE_H_
#define METADEF_CXX_INC_EXE_GRAPH_STORAGE_SHAPE_H_
#include <type_traits>
#include "shape.h"

namespace gert {
struct StorageShape {
 public:
  StorageShape() {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  };
  /**
   * 构造一个运行时shape实例
   * @param origin_shape 原始shape
   * @param storage_shape 运行时shape
   */
  StorageShape(const std::initializer_list<int64_t> &origin_shape, const std::initializer_list<int64_t> &storage_shape)
      : origin_shape_(origin_shape), storage_shape_(storage_shape) {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }
  /**
   * 获取原始shape
   * @return 原始shape
   */
  const Shape &GetOriginShape() const {
    return origin_shape_;
  }
  /**
   * 获取运行时shape
   * @return 运行时shape
   */
  const Shape &GetStorageShape() const {
    return storage_shape_;
  }
  /**
   * 获取可写的原始shape
   * @return 可写的原始shape
   */
  Shape &MutableOriginShape() {
    return origin_shape_;
  }
  /**
   * 获取可写的运行时shape
   * @return 可写的运行时shape
   */
  Shape &MutableStorageShape() {
    return storage_shape_;
  }
  /**
   * 判断shape是否相等
   * @param other 另一个shape
   * @return true表示相等
   */
  bool operator==(const StorageShape &other) const {
    return origin_shape_ == other.origin_shape_ && storage_shape_ == other.storage_shape_;
  }
  /**
   * 判断shape是否不相等
   * @param other 另一个shape
   * @return true表示不相等
   */
  bool operator!=(const StorageShape &other) const {
    return !(*this == other);
  }
 private:
  Shape origin_shape_;
  Shape storage_shape_;
  uint8_t reserved_[40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<StorageShape>::value, "The class must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_STORAGE_SHAPE_H_
