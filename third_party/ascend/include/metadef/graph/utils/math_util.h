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

#ifndef METADEF_CXX_INC_GRAPH_UTILS_MATH_UTIL_H_
#define METADEF_CXX_INC_GRAPH_UTILS_MATH_UTIL_H_

#include <securec.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <iostream>

#include "graph/def_types.h"
#include "external/utils/extern_math_util.h"
#include "common/ge_common/debug/log.h"

namespace ge {
constexpr uint32_t kDiv16RightShiftBits = 4U;
constexpr uint32_t kDiv32RightShiftBits = 5U;
/**
 * @ingroup domi_calibration
 * @brief  Initializes an input array to a specified value
 * @param [in]  n        array initialization length
 * @param [in]  alpha    initialization value
 * @param [out]  output  array to be initialized
 * @return      Status
 */
template<typename Dtype>
Status NnSet(const int32_t n, const Dtype alpha, Dtype *const output) {
  GE_CHECK_NOTNULL(output);

  if (std::equal_to<Dtype>()(alpha, static_cast<Dtype>(0))) {
    if ((sizeof(Dtype) * static_cast<size_t>(n)) < SECUREC_MEM_MAX_LEN) {
      const errno_t err =
          memset_s(output, sizeof(Dtype) * static_cast<size_t>(n), 0, sizeof(Dtype) * static_cast<size_t>(n));
      GE_CHK_BOOL_RET_STATUS(err == EOK, PARAM_INVALID, "memset_s err");
    } else {
      const uint64_t size = static_cast<uint64_t>(sizeof(Dtype) * static_cast<size_t>(n));
      const uint64_t step = SECUREC_MEM_MAX_LEN - (SECUREC_MEM_MAX_LEN % sizeof(Dtype));
      const uint64_t times = size / step;
      const uint64_t remainder = size % step;
      uint64_t i = 0U;
      while (i < times) {
        const errno_t err = memset_s(ValueToPtr(PtrToValue(output) + (i * (step / sizeof(Dtype)))), step, 0, step);
        GE_CHK_BOOL_RET_STATUS(err == EOK, PARAM_INVALID, "memset_s err");
        i++;
      }
      if (remainder != 0U) {
        const errno_t err =
            memset_s(ValueToPtr(PtrToValue(output) + (i * (step / sizeof(Dtype)))), remainder, 0, remainder);
        GE_CHK_BOOL_RET_STATUS(err == EOK, PARAM_INVALID, "memset_s err");
      }
    }
  }

  for (int32_t i = 0; i < n; ++i) {
    output[i] = alpha;
  }
  return SUCCESS;
}

template<typename T, typename TR>
bool RoundUpOverflow(T value, T multiple_of, TR &ret) {
  if (multiple_of == 0) {
    ret = 0;
    return true;
  }
  auto remainder = value % multiple_of;
  if (remainder == 0) {
    if (!IntegerChecker<TR>::Compat(value)) {
      return true;
    }
    ret = static_cast<TR>(value);
    return false;
  }
  return AddOverflow(value - remainder, multiple_of, ret);
}
template<typename T>
T CeilDiv16(const T n) {
  if (n & 0xF) {
    return (n >> kDiv16RightShiftBits) + 1;
  } else {
    return n >> kDiv16RightShiftBits;
  }
}

template<typename T>
T CeilDiv32(const T n) {
  if (n & 0x1F) {
    return (n >> kDiv32RightShiftBits) + 1;
  } else {
    return n >> kDiv32RightShiftBits;
  }
}

template<typename T>
T CeilDiv(const T n1, const T n2) {
  if (n1 == 0) {
    return 0;
  }
  return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template<typename T>
T FloorDiv(const T u_value, const T d_value) {
  if (d_value == 0) {
    return u_value;
  }
  return u_value / d_value;
}

inline uint64_t RoundUp(const uint64_t origin_value, const uint64_t multiple_of) {
  uint64_t ret = 0U;
  if (RoundUpOverflow(origin_value, multiple_of, ret)) {
    return 0U;
  }
  return ret;
}

inline Status GeMemcpy(uint8_t *dst_ptr, size_t dst_size, const uint8_t *src_ptr, const size_t src_size) {
  GE_CHK_BOOL_RET_STATUS((dst_size >= src_size), PARAM_INVALID, "memcpy_s verify fail, src size %zu, dst size %zu",
      src_size, dst_size);
  size_t offset = 0U;
  size_t remain_size = src_size;
  do {
    size_t copy_size = (remain_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : remain_size;
    const auto err = memcpy_s((dst_ptr + offset), copy_size, (src_ptr + offset), copy_size);
    GE_CHK_BOOL_RET_STATUS(err == EOK, PARAM_INVALID, "memcpy_s err, src size %zu, dst size %zu",
        copy_size, (dst_size - offset));
    offset += copy_size;
    remain_size -= copy_size;
  } while (remain_size > 0U);
  return SUCCESS;
}
}  // end namespace ge

#define REQUIRE_COMPAT(T, v)                                                                                           \
  do {                                                                                                                 \
    if (!IntegerChecker<T>::Compat((v))) {                                                                             \
      std::stringstream ss;                                                                                            \
      ss << #v << " value " << (v) << " out of " << #T << " range [" << std::numeric_limits<T>::min() << ","           \
         << std::numeric_limits<T>::max() << "]";                                                                      \
      GELOGE(ge::FAILED, "%s", ss.str().c_str());                                                                      \
      return ge::FAILED;                                                                                               \
    }                                                                                                                  \
  } while (false)

#define REQUIRE_COMPAT_FOR_CHAR(T, v)                                                                                  \
  do {                                                                                                                 \
    if (!IntegerChecker<T>::Compat((v))) {                                                                             \
      std::stringstream ss;                                                                                            \
      ss << #v << " value " << static_cast<int32_t>(v) << " out of " << #T << " range ["                               \
         << static_cast<int32_t>(std::numeric_limits<T>::min()) << ","                                                 \
         << static_cast<int32_t>(std::numeric_limits<T>::max()) << "]";                                                \
      GELOGE(ge::FAILED, "%s", ss.str().c_str());                                                                      \
      return ge::FAILED;                                                                                               \
    }                                                                                                                  \
  } while (false)

#define REQUIRE_COMPAT_INT8(v) REQUIRE_COMPAT_FOR_CHAR(int8_t, (v))
#define REQUIRE_COMPAT_UINT8(v) REQUIRE_COMPAT_FOR_CHAR(uint8_t, (v))
#define REQUIRE_COMPAT_INT16(v) REQUIRE_COMPAT(int16_t, (v))
#define REQUIRE_COMPAT_UINT16(v) REQUIRE_COMPAT(uint16_t, (v))
#define REQUIRE_COMPAT_INT32(v) REQUIRE_COMPAT(int32_t, (v))
#define REQUIRE_COMPAT_UINT32(v) REQUIRE_COMPAT(uint32_t, (v))
#define REQUIRE_COMPAT_INT64(v) REQUIRE_COMPAT(int64_t, (v))
#define REQUIRE_COMPAT_UINT64(v) REQUIRE_COMPAT(uint64_t, (v))
#define REQUIRE_COMPAT_SIZE_T(v) REQUIRE_COMPAT(size_t, (v))

#endif  // METADEF_CXX_INC_GRAPH_UTILS_MATH_UTIL_H_
