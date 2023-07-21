#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_CHECKER_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_CHECKER_H_

#include <vector>
#include <cstdarg>
#include <cstdio>
#include "securec.h"

#include "external/graph/types.h"
#include "tng_status.h"
#include "external/utils/extern_math_util.h"
#include "compat_apis.h"
#include "logger.h"

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

inline std::vector<char> CreateErrorMsg() { return {}; }

#define TNG_ASSERT_EQ(x, y)                                                                    \
  do {                                                                                         \
    const auto &xv = (x);                                                                      \
    const auto &yv = (y);                                                                      \
    if (xv != yv) {                                                                            \
      std::stringstream ss;                                                                    \
      ss << "Assert (" << #x << " == " << #y << ") failed, expect " << yv << " actual " << xv; \
      return tng::Status::Error(ss.str().c_str());                                             \
    }                                                                                          \
  } while (false)

#define TNG_ASSERT(exp, ...)                                 \
  do {                                                       \
    if (!(exp)) {                                            \
      auto msg = CreateErrorMsg(__VA_ARGS__);                \
      if (msg.empty()) {                                     \
        return tng::Status::Error("Assert %s failed", #exp); \
      }                                                      \
      return tng::Status::Error("%s", msg.data());           \
    }                                                        \
  } while (false)

#define TNG_ASSERT_NOTNULL(v, ...) TNG_ASSERT(((v) != nullptr), __VA_ARGS__)

#define TNG_RETURN_IF_ERROR(expr)                 \
  do {                                            \
    const auto &status = (expr);                  \
    if (!status.IsSuccess()) {                    \
      return status;                              \
    }                                             \
  } while (false)

#define TNG_ASSERT_GE_OK(expr)             \
  do {                                     \
    const auto &status = (expr);           \
    if (status != ge::SUCCESS) {           \
      return tng::compat::GeErrorStatus(); \
    }                                      \
  } while (false)

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_CHECKER_H_