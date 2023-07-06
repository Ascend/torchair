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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTINUOUS_BUFFER_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTINUOUS_BUFFER_H_
#include <type_traits>
#include <cstdint>
#include "graph/def_types.h"
namespace gert {
namespace bg {
class BufferPool;
}
class ContinuousBuffer {
 public:
  /**
   * 获取buffer的数量
   * @return buffer的数量
   */
  size_t GetNum() const {
    return num_;
  }
  /**
   * 获取本实例的总长度
   * @return 本实例的总长度，单位为字节
   */
  size_t GetTotalLength() const {
    return offsets_[num_];
  }
  /**
   * 获取一个buffer
   * @tparam T buffer的类型
   * @param index buffer的index
   * @return 指向该buffer的指针，若index非法，则返回空指针
   */
  template<typename T>
  const T *Get(size_t index) const {
    if (index >= num_) {
      return nullptr;
    }
    return ge::PtrToPtr<uint8_t, T>(ge::PtrToPtr<ContinuousBuffer, uint8_t>(this) + offsets_[index]);
  }
  /**
   * 获取一个buffer，及其对应的长度
   * @tparam T buffer的类型
   * @param index buffer的index
   * @param len buffer的长度
   * @return 指向该buffer的指针，若index非法，则返回空指针
   */
  template<typename T>
  const T *Get(size_t index, size_t &len) const {
    if (index >= num_) {
      return nullptr;
    }
    len = offsets_[index + 1] - offsets_[index];
    return ge::PtrToPtr<uint8_t, T>(ge::PtrToPtr<ContinuousBuffer, uint8_t>(this) + offsets_[index]);
  }
  /**
   * 获取一个buffer
   * @tparam T buffer的类型
   * @param index buffer的index
   * @return 指向该buffer的指针，若index非法，则返回空指针
   */
  template<typename T>
  T *Get(size_t index) {
    if (index >= num_) {
      return nullptr;
    }
    return ge::PtrToPtr<uint8_t, T>(ge::PtrToPtr<ContinuousBuffer, uint8_t>(this) + offsets_[index]);
  }

 private:
  friend ::gert::bg::BufferPool;
  size_t num_;
  int64_t reserved_; // Reserved field, 8-byte aligned
  size_t offsets_[1];
};
static_assert(std::is_standard_layout<ContinuousBuffer>::value, "The class ContinuousText must be POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTINUOUS_BUFFER_H_
