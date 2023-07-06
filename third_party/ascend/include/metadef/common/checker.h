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

#ifndef METADEF_CXX_INC_COMMON_CHECKER_H_
#define METADEF_CXX_INC_COMMON_CHECKER_H_
#include <securec.h>
#include <sstream>
#include "graph/ge_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "hyper_status.h"

struct ErrorResult {
  operator bool() const {
    return false;
  }
  operator ge::graphStatus() const {
    return ge::PARAM_INVALID;
  }
  template<typename T>
  operator std::unique_ptr<T>() const {
    return nullptr;
  }
  template<typename T>
  operator std::shared_ptr<T>() const {
    return nullptr;
  }
  template<typename T>
  operator T *() const {
    return nullptr;
  }
  template<typename T>
  operator std::vector<std::shared_ptr<T>>() const {
    return {};
  }
  template<typename T>
  operator std::vector<T>() const {
    return {};
  }
};

inline std::vector<char> CreateErrorMsg(const char *format, ...) {
  va_list args;
  va_start(args, format);
  va_list args_copy;
  va_copy(args_copy, args);
  const size_t len = static_cast<size_t>(vsnprintf(nullptr, 0, format, args_copy));
  va_end(args_copy);
  std::vector<char> msg(len + 1U, '\0');
  const auto ret = vsnprintf_s(msg.data(), len + 1U, len, format, args);
  va_end(args);
  return (ret > 0) ? msg : std::vector<char>{};
}

inline std::vector<char> CreateErrorMsg() {
  return {};
}

#define GE_ASSERT_EQ(x, y)                                                                                             \
  do {                                                                                                                 \
    const auto &xv = (x);                                                                                              \
    const auto &yv = (y);                                                                                              \
    if (xv != yv) {                                                                                                    \
      std::stringstream ss;                                                                                            \
      ss << "Assert (" << #x << " == " << #y << ") failed, expect " << yv << " actual " << xv;                         \
      REPORT_INNER_ERROR("E19999", "%s", ss.str().c_str());                                                            \
      GELOGE(ge::FAILED, "%s", ss.str().c_str());                                                                      \
      return ::ErrorResult();                                                                                          \
    }                                                                                                                  \
  } while (false)

#define GE_ASSERT(exp, ...)                                                                                            \
  do {                                                                                                                 \
    if (!(exp)) {                                                                                                      \
      auto msg = CreateErrorMsg(__VA_ARGS__);                                                                          \
      if (msg.empty()) {                                                                                               \
        REPORT_INNER_ERROR("E19999", "Assert %s failed", #exp);                                                        \
        GELOGE(ge::FAILED, "Assert %s failed", #exp);                                                                  \
      } else {                                                                                                         \
        REPORT_INNER_ERROR("E19999", "%s", msg.data());                                                                \
        GELOGE(ge::FAILED, "%s", msg.data());                                                                          \
      }                                                                                                                \
      return ::ErrorResult();                                                                                          \
    }                                                                                                                  \
  } while (false)

#define GE_ASSERT_NOTNULL(v, ...) GE_ASSERT(((v) != nullptr), __VA_ARGS__)
#define GE_ASSERT_SUCCESS(v, ...) GE_ASSERT(((v) == ge::SUCCESS), __VA_ARGS__)
#define GE_ASSERT_GRAPH_SUCCESS(v, ...) GE_ASSERT(((v) == ge::GRAPH_SUCCESS), __VA_ARGS__)
#define GE_ASSERT_RT_OK(v, ...) GE_ASSERT(((v) == 0), __VA_ARGS__)
#define GE_ASSERT_EOK(v, ...) GE_ASSERT(((v) == EOK), __VA_ARGS__)
#define GE_ASSERT_TRUE(v, ...) GE_ASSERT((v), __VA_ARGS__)
#define GE_ASSERT_HYPER_SUCCESS(v, ...) GE_ASSERT(((v).IsSuccess()), __VA_ARGS__)


#define GE_RETURN_IF(exp, ...)                                                                                         \
  do {                                                                                                                 \
    if (exp) {                                                                                                         \
      auto msg = CreateErrorMsg(__VA_ARGS__);                                                                          \
      if (msg.empty()) {                                                                                               \
        REPORT_INNER_ERROR("E19999", "Assert %s failed", #exp);                                                        \
        GELOGE(ge::FAILED, "Assert %s failed", #exp);                                                                  \
      } else {                                                                                                         \
        REPORT_INNER_ERROR("E19999", "%s", msg.data());                                                                \
        GELOGE(ge::FAILED, "%s", msg.data());                                                                          \
      }                                                                                                                \
      return;                                                                                                          \
    }                                                                                                                  \
  } while (false)

#define GE_RETURN_IF_NULL(v, ...) GE_RETURN_IF(((v) == nullptr), __VA_ARGS__)
#define GE_RETURN_IF_SUCCESS(v, ...) GE_RETURN_IF(((v) == ge::SUCCESS), __VA_ARGS__)
#define GE_RETURN_IF_GRAPH_SUCCESS(v, ...) GE_RETURN_IF(((v) == ge::GRAPH_SUCCESS), __VA_ARGS__)
#define GE_RETURN_IF_RT_OK(v, ...) GE_RETURN_IF(((v) == 0), __VA_ARGS__)
#define GE_RETURN_IF_EOK(v, ...) GE_RETURN_IF(((v) == EOK), __VA_ARGS__)
#define GE_RETURN_IF_TRUE(v, ...) GE_RETURN_IF((v), __VA_ARGS__)
#define GE_RETURN_IF_HYPER_SUCCESS(v, ...) GE_RETURN_IF(((v).IsSuccess()), __VA_ARGS__)
#endif  // METADEF_CXX_INC_COMMON_CHECKER_H_
