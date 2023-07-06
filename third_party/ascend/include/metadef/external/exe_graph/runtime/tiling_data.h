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
#ifndef METADEF_CXX_INC_EXE_GRAPH_TILING_DATA_H_
#define METADEF_CXX_INC_EXE_GRAPH_TILING_DATA_H_
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <securec.h>
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "graph/ge_error_codes.h"
#include "utils/extern_math_util.h"

namespace gert {
enum class AttrDataType : int32_t {
  kBool = 0,
  kString,
  kInt32,
  kInt64,
  kUint32,
  kFloat32,
  kFloat16,
  kListBool,
  kListString,
  kListInt32,
  kListInt64,
  kListUint32,
  kListFloat32,
  kListFloat16,
  kListListInt32,
  kListListInt64,
  kTypeEnd
};
class TilingData {
 public:
  /**
   * 获取本实例可容纳的最大tiling data长度
   * @return 最大tiling data长度
   */
  size_t GetCapacity() const {
    return capacity_;
  }
  /**
   * 获取tiling data长度
   * @return tiling data长度
   */
  size_t GetDataSize() const {
    return data_size_;
  }
  /**
   * 设置tiling data长度
   * @param size tiling data长度
   */
  void SetDataSize(const size_t size) {
    data_size_ = size;
  }
  /**
   * 获取data指针
   * @return data指针
   */
  void *GetData() {
    return data_;
  }
  /**
   * 获取data指针
   * @return data指针
   */
  const void *GetData() const {
    return data_;
  }
  /**
   * 向后添加tiling data，若添加超过可容纳的最大长度，则添加失败
   * @tparam T 添加的tiling data的类型
   * @param data 添加的tiling data实例
   * @return 成功返回ge::GRAPH_SUCCESS
   */
  template<typename T, typename std::enable_if<std::is_standard_layout<T>::value, int>::type = 0>
  ge::graphStatus Append(const T &data) {
    size_t after_size;
    if (ge::AddOverflow(data_size_, sizeof(data), after_size)) {
      return ge::GRAPH_FAILED;
    }
    if (after_size > capacity_) {
      return ge::GRAPH_FAILED;
    }
    *reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(data_) + GetDataSize()) = data;
    data_size_ = after_size;
    return ge::GRAPH_SUCCESS;
  }

  template<typename T, typename std::enable_if<std::is_standard_layout<T>::value, int>::type = 0>
  ge::graphStatus Append(const T *data, size_t append_num) {
    size_t append_size;
    if (ge::MulOverflow(sizeof(T), append_num, append_size)) {
      return ge::GRAPH_MUL_OVERFLOW;
    }
    size_t after_size;
    if (ge::AddOverflow(data_size_, append_size, after_size)) {
      return ge::GRAPH_ADD_OVERFLOW;
    }
    const auto ret =
        memcpy_s(reinterpret_cast<uint8_t *>(data_) + data_size_, capacity_ - data_size_, data, append_size);
    if (ret != EOK) {
      return ge::GRAPH_MEMCPY_FAILED;
    }
    data_size_ = after_size;
    return ge::GRAPH_SUCCESS;
  }

  /**
   * 通过最大容量创建一个TilingData类实例
   * @param cap_size 最大容量，单位为字节
   * @return 实例指针
   */
  static std::unique_ptr<uint8_t[]> CreateCap(const size_t cap_size) {
    size_t total_size;
    if (ge::AddOverflow(sizeof(TilingData), cap_size, total_size)) {
      return nullptr;
    }
    auto td_buf = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]());
    if (td_buf == nullptr) {
      return nullptr;
    }
    const auto td = reinterpret_cast<TilingData *>(td_buf.get());
    td->Init(cap_size, td_buf.get() + sizeof(TilingData));
    return td_buf;
  }
  /**
   * 通过最大容量计算TilingData实例所占用的内存空间
   * @param cap_size 最大容量，单位为字节
   * @param total_size 内存空间，单位为字节
   * @return 成功返回ge::GRAPH_SUCCESS
   */
  static ge::graphStatus CalcTotalSize(const size_t cap_size, size_t &total_size) {
    if (ge::AddOverflow(sizeof(TilingData), cap_size, total_size)) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 初始化TilingData
   * @param cap_size 最大容量
   * @param data tiling data的地址
   */
  void Init(const size_t cap_size, void *const data) {
    capacity_ = cap_size;
    data_size_ = 0UL;
    data_ = data;
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }

  ge::graphStatus AppendConvertedAttrVal(const RuntimeAttrs *attrs, const size_t attr_index,
                                         const AttrDataType src_type, const AttrDataType dst_type);
  TilingData(const TilingData &) = delete;
  TilingData(TilingData &&) = delete;
  TilingData operator=(const TilingData &) = delete;
  TilingData operator=(TilingData &&) = delete;

 private:
  TilingData() = default;
  size_t capacity_;
  size_t data_size_;
  void *data_;
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
};

/**
 * 向后添加tiling data，若添加超过可容纳的最大长度，则忽略本次操作
 * @tparam T 添加的tiling data的类型
 * @param out TilingData类实例
 * @param data 添加的tiling data的实例
 * @return
 */
template<typename T>
TilingData &operator<<(TilingData &out, const T &data) {
  out.Append(data);  // we can not throw exception, so callers can not get the error information
  return out;
}
static_assert(std::is_standard_layout<TilingData>::value, "The class TilingData must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_TILING_DATA_H_
