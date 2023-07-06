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

#ifndef METADEF_CXX_INC_EXE_GRAPH_STORAGE_FORMAT_H_
#define METADEF_CXX_INC_EXE_GRAPH_STORAGE_FORMAT_H_
#include <securec.h>
#include <memory>
#include "graph/types.h"
#include "expand_dims_type.h"
namespace gert {
struct StorageFormat {
 public:
  StorageFormat() {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  };
  /**
   * 构造一个格式，格式包括原始格式、运行时格式、补维规则
   * @param origin_format 原始格式
   * @param storage_format 运行时格式
   * @param expand_dims_type 补维规则
   */
  StorageFormat(const ge::Format origin_format, const ge::Format storage_format, const ExpandDimsType &expand_dims_type)
      : origin_format_(origin_format), storage_format_(storage_format), expand_dims_type_(expand_dims_type) {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }
  /**
   * 获取原始format
   * @return 原始format
   */
  ge::Format GetOriginFormat() const {
    return origin_format_;
  }
  /**
   * 设置原始format
   * @param origin_format 原始format
   */
  void SetOriginFormat(const ge::Format origin_format) {
    origin_format_ = origin_format;
  }
  /**
   * 获取运行时format
   * @return 运行时format
   */
  ge::Format GetStorageFormat() const {
    return storage_format_;
  }
  /**
   * 设置运行时format
   * @param storage_format 运行时format
   */
  void SetStorageFormat(const ge::Format storage_format) {
    storage_format_ = storage_format;
  }
  /**
   * 获取补维规则
   * @return 补维规则
   */
  ExpandDimsType GetExpandDimsType() const {
    return expand_dims_type_;
  }
  /**
   * 设置补维规则
   * @param expand_dims_type 补维规则
   */
  void SetExpandDimsType(const ExpandDimsType &expand_dims_type) {
    expand_dims_type_ = expand_dims_type;
  }
  /**
   * 获取可写的补维规则
   * @return 补维规则引用
   */
  ExpandDimsType &MutableExpandDimsType() {
    return expand_dims_type_;
  }
  /**
   * 判断格式是否相等
   * @param other 另一个格式
   * @return true代表相等
   */
  bool operator==(const StorageFormat &other) const {
    return origin_format_ == other.origin_format_ && storage_format_ == other.storage_format_ &&
        expand_dims_type_ == other.expand_dims_type_;
  }
  /**
   * 判断格式是否不相等
   * @param other 另一个格式
   * @return true代表不相等
   */
  bool operator!=(const StorageFormat &other) const {
    return !(*this == other);
  }

 private:
  ge::Format origin_format_;
  ge::Format storage_format_;
  ExpandDimsType expand_dims_type_;
  uint8_t reserved_[40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<StorageFormat>::value, "The class StorageFormat must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_STORAGE_FORMAT_H_
