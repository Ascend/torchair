/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef METADEF_CXX_INC_EXTERNAL_UTILS_EXTERN_MATH_UTIL_H
#define METADEF_CXX_INC_EXTERNAL_UTILS_EXTERN_MATH_UTIL_H

#include <iostream>
#include <limits>

namespace ge {
template<typename T>
class IntegerChecker {
 public:
  template<typename T1>
  static bool Compat(const T1 v) {
    static_assert(((sizeof(T) <= sizeof(uint64_t)) && (sizeof(T1) <= sizeof(uint64_t))),
                  "IntegerChecker can only check integers less than 64 bits");
    if (v >= static_cast<T1>(0)) {
      return static_cast<uint64_t>(v) <= static_cast<uint64_t>(std::numeric_limits<T>::max());
    }
    return static_cast<int64_t>(v) >= static_cast<int64_t>(std::numeric_limits<T>::min());
  }
};

template<typename TLhs, typename TRhs, typename TRet>
bool MulOverflow(TLhs lhs, TRhs rhs, TRet &ret) {
#if __GNUC__ >= 5
  return __builtin_mul_overflow(lhs, rhs, &ret);
#else
  if ((!IntegerChecker<TRet>::Compat(lhs)) || (!IntegerChecker<TRet>::Compat(rhs))) {
    return true;
  }
  if ((lhs == 0) || (rhs == 0)) {
    ret = 0;
    return false;
  }
  TRet reminder = std::numeric_limits<TRet>::max() / static_cast<TRet>(rhs);
  const TRet lhs_ret_type = static_cast<TRet>(lhs);
  if (lhs_ret_type < 0) {
    if (reminder > 0) {
      reminder *= static_cast<TRet>(-1);
    }
    if (lhs_ret_type < reminder) {
      return true;
    }
  } else {
    if (reminder < 0) {
      reminder *= static_cast<TRet>(-1);
    }
    if (lhs_ret_type > reminder) {
      return true;
    }
  }
  ret = static_cast<TRet>(lhs) * static_cast<TRet>(rhs);
  return false;
#endif
}

template<typename TLhs, typename TRhs, typename TRet>
bool AddOverflow(TLhs lhs, TRhs rhs, TRet &ret) {
#if __GNUC__ >= 5
  return __builtin_add_overflow(lhs, rhs, &ret);
#else
  if ((!IntegerChecker<TRet>::Compat(lhs)) || (!IntegerChecker<TRet>::Compat(rhs))) {
    return true;
  }
  if (rhs >= 0) {
    if (static_cast<TRet>(lhs) > std::numeric_limits<TRet>::max() - static_cast<TRet>(rhs)) {
      return true;
    }
  } else {
    if (static_cast<TRet>(lhs) < std::numeric_limits<TRet>::min() - static_cast<TRet>(rhs)) {
      return true;
    }
  }
  ret = static_cast<TRet>(lhs) + static_cast<TRet>(rhs);
  return false;
#endif
}
} // namespace ge


#endif  // METADEF_CXX_INC_EXTERNAL_UTILS_EXTERN_MATH_UTIL_H
