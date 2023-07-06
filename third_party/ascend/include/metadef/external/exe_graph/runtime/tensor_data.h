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
#ifndef METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_H_
#define METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_H_

#include <securec.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "graph/ge_error_codes.h"

namespace gert {
using TensorAddress = void *;
using ConstTensorAddressPtr = const void *;

enum TensorPlacement : int32_t {
    kOnDeviceHbm,  ///< Tensor位于Device上的HBM内存
    kOnHost,       ///< Tensor位于Host
    kFollowing,    ///< Tensor位于Host，且数据紧跟在结构体后面
    kTensorPlacementEnd
};
inline const char *GetPlacementStr(const TensorPlacement placement) {
  static const char
      *placement_str[static_cast<int32_t>(kTensorPlacementEnd) + 1] = {
          "DeviceHbm", "HostDDR", "HostDDR", "Unknown"};
  if ((placement >= kTensorPlacementEnd) || (placement < kOnDeviceHbm)) {
    return placement_str[kTensorPlacementEnd];
  }
  return placement_str[placement];
}

enum TensorOperateType : int32_t {
  kGetTensorAddress,  ///< 获取Tensor的地址
  kFreeTensor,        ///< 释放Tensor
  kPlusShareCount,    ///< 共享Tensor
  kTensorOperateType
};

/**
 * Tensor的管理函数
 */
using TensorAddrManager = ge::graphStatus (*)(TensorAddress addr, TensorOperateType operate_type, void **out);

class TensorData {
 public:
  /**
   * 构造一个TensorData
   * @param addr tensor的地址
   * @param manager tensor data的管理函数，若manager为空，则认为addr就是tensor的数据地址，且此数据不需要被释放
   */
  explicit TensorData(TensorAddress addr = nullptr, const TensorAddrManager manager = nullptr)
      : addr_(addr), manager_(manager), size_(0U), placement_(kTensorPlacementEnd), reserved_0_(0U) {
    (void)memset_s(reserved_1_, sizeof(reserved_1_), 0, sizeof(reserved_1_));
  }
  explicit TensorData(TensorAddress addr, const TensorAddrManager manager, size_t size, TensorPlacement placement)
      : addr_(addr), manager_(manager), size_(size), placement_(placement), reserved_0_(0U) {
    (void)memset_s(reserved_1_, sizeof(reserved_1_), 0, sizeof(reserved_1_));
  }
  TensorData(const TensorData &) = delete;
  TensorData(TensorData &&other) noexcept : addr_(other.addr_), manager_(other.manager_),
    size_(other.size_), placement_(other.placement_) {
    other.addr_ = nullptr;
    other.manager_ = nullptr;
    other.size_ = 0U;
    other.placement_ = kTensorPlacementEnd;
    reserved_0_ = 0U;
    (void)memset_s(reserved_1_, sizeof(reserved_1_), 0, sizeof(reserved_1_));
  }
  TensorData &operator=(const TensorData &other) = delete;
  TensorData &operator=(TensorData &&other) noexcept {
    if (this != &other) {
      static_cast<void>(Free());
      addr_ = other.addr_;
      manager_ = other.manager_;
      size_ = other.size_;
      placement_ = other.placement_;
      other.addr_ = nullptr;
      other.manager_ = nullptr;
      other.size_ = 0U;
      other.placement_ = kTensorPlacementEnd;
      reserved_0_ = 0U;
      (void)memset_s(reserved_1_, sizeof(reserved_1_), 0, sizeof(reserved_1_));
    }
    return *this;
  }
  ~TensorData() noexcept {
    static_cast<void>(Free());
  }

  /**
   * 获取tensor地址
   * @return tensor地址
   */
  TensorAddress GetAddr() const {
    if (manager_ == nullptr || addr_ == nullptr) {
      return addr_;
    }
    TensorAddress addr;
    if (manager_(addr_, kGetTensorAddress, &addr) != ge::GRAPH_SUCCESS) {
      return nullptr;
    } else {
      return addr;
    }
  }

  /**
  * 获取tensor的内存大小
  * @return tensor所占内存大小
  */
  size_t GetSize() const {
    return size_;
  }
  /**
   * 设置tensor的内存大小
   * @param tensor的内存大小
   */
  void SetSize(const size_t size) {
    size_ = size;
  }

  /**
  * 获取tensor的placement
  * @return tensor的placement
  */
  TensorPlacement GetPlacement() const {
    return placement_;
  }
  /**
   * 设置tensor的placement
   * @param tensor的placement
   */
  void SetPlacement(const TensorPlacement placement) {
    placement_ = placement;
  }
  /**
   * 释放tensor
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus Free() {
    if (manager_ == nullptr) {
      return ge::GRAPH_SUCCESS;
    }
    const auto ret = manager_(addr_, kFreeTensor, nullptr);
    if (ret == ge::GRAPH_SUCCESS) {
      addr_ = nullptr;
      manager_ = nullptr;
    }
    return ret;
  }
  /**
   * 设置tensor地址
   * @param addr tensor地址
   * @param manager tensor的管理函数
   */
  ge::graphStatus SetAddr(const ConstTensorAddressPtr addr, TensorAddrManager manager) {
    const auto ret = Free();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    addr_ = const_cast<TensorAddress>(addr);
    manager_ = manager;
    return ge::GRAPH_SUCCESS;
  }

  bool IsSharedWith(const TensorData &other) const {
    return (addr_ == other.addr_ && manager_ == other.manager_ && size_ == other.size_ &&
            placement_ == other.placement_);
  }

  ge::graphStatus ShareFrom(const TensorData &other) {
    if (IsSharedWith(other)) {
      return ge::GRAPH_SUCCESS;
    }
    const auto ret = Free();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }

    addr_ = other.addr_;
    manager_ = other.manager_;
    size_ = other.size_;
    placement_ = other.placement_;
    if (manager_ != nullptr) {
      return manager_(addr_, kPlusShareCount, nullptr);
    } else {
      return ge::GRAPH_SUCCESS;
    }
  }

 private:
  TensorAddress addr_;
  TensorAddrManager manager_;
  size_t size_;
  TensorPlacement placement_;
  uint32_t reserved_0_;  // Reserved field, 8-byte aligned for TensorPlacement
  uint8_t reserved_1_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
};
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_H_
