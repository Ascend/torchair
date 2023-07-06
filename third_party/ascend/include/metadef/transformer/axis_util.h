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

#ifndef COMMON_UTILS_TRANSFORMER_INC_AXIS_UTIL_H_
#define COMMON_UTILS_TRANSFORMER_INC_AXIS_UTIL_H_

#include <memory.h>
#include <array>

#include "external/graph/types.h"
#include "graph/utils/math_util.h"
#include "exe_graph/runtime/shape.h"

namespace transformer {
#define CHECK(cond, log_func, return_expr) \
  do {                                     \
    if (cond) {                            \
      log_func;                            \
      return_expr;                         \
    }                                      \
  } while (0)

#define INT64_ZEROCHECK(a)                 \
  if (a == 0) {                            \
    return false;                          \
  }

#define MUL_OVERFLOW(x, y, z)             \
  if (ge::MulOverflow((x), (y), (z))) {   \
    return false;                         \
  }                                       \

enum AxisValueType {
  AXIS_N = 0,
  AXIS_C = 1,
  AXIS_H = 2,
  AXIS_W = 3,
  AXIS_C1 = 4,
  AXIS_C0 = 5,
  AXIS_Co = 6,
  AXIS_D = 7,
  AXIS_G = 8,
  AXIS_M0 = 9,
  AXIS_INPUT_SIZE = 10,
  AXIS_HIDEEN_SIZE = 11,
  AXIS_STATE_SIZE = 12,
  AXIS_BOTTOM = 13
};

using AxisValue = std::array<int64_t, static_cast<size_t>(AXIS_BOTTOM)>;

inline int64_t DivisionCeiling(int64_t dividend, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  } else if (dividend < 0) {
    return -1;
  } else {
    return (dividend + divisor - 1) / divisor;
  }
}

class AxisUtil {
 public:
  AxisUtil() {};
  ~AxisUtil() {};
  static bool GetAxisValueByOriginFormat(const ge::Format &format, const gert::Shape &shape, AxisValue &axis_value);

private:
  static bool GetAxisValueByNCHW(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByNHWC(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByHWCN(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByND(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByNDHWC(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByNCDHW(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByDHWCN(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByDHWNC(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByNC1HWC0(const gert::Shape &shape, AxisValue &axis_value);

  static bool GetAxisValueByC1HWNCoC0(const gert::Shape &shape, AxisValue &axis_value);
};
} // namespace transformer
#endif // COMMON_UTILS_TRANSFORMER_INC_AXIS_UTIL_H_
