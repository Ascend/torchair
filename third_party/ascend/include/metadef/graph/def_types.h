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

#ifndef INC_GRAPH_DEF_TYPES_H_
#define INC_GRAPH_DEF_TYPES_H_

#include <atomic>
#include <memory>
#include <vector>
namespace ge {
// used for data type of DT_STRING
struct StringHead {
  int64_t addr;  // the addr of string
  int64_t len;   // the length of string
};

inline uint64_t PtrToValue(const void *const ptr) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

inline void *ValueToPtr(const uint64_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}

inline std::vector<uint64_t> VPtrToValue(const std::vector<void *> v_ptr) {
  std::vector<uint64_t> v_value;
  for (const auto &ptr : v_ptr) {
    v_value.emplace_back(PtrToValue(ptr));
  }
  return v_value;
}

template<typename TI, typename TO>
inline TO *PtrToPtr(TI *const ptr) {
  return reinterpret_cast<TO *>(ptr);
}

template<typename TI, typename TO>
inline const TO *PtrToPtr(const TI *const ptr) {
  return reinterpret_cast<const TO *>(ptr);
}

template<typename T>
inline T *PtrAdd(T *const ptr, const size_t max_buf_len, const size_t idx) {
  if ((ptr != nullptr) && (idx < max_buf_len)) {
    return reinterpret_cast<T *>(ptr + idx);
  }
  return nullptr;
}
}  // namespace ge

#endif  // INC_GRAPH_DEF_TYPES_H_
