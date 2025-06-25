#include "tng_status.h"

#include <cstring>
#include <string>
#include <regex>
#include <memory>
#include <securec.h>
#include <set>

#include "logger.h"
#include "external/graph/types.h"
#include "external/graph/ascend_string.h"

namespace tng {
static const std::set<char> valid_log_levels = {'0', '1', '2', '3', '4'};

int32_t Logger::kLogLevel = []() -> int32_t {
  auto env_val = std::getenv("TNG_LOG_LEVEL");
  if (env_val) {
    std::string env_tmp(env_val);
    if (env_tmp.empty()) {
      return static_cast<int32_t>(tng::LogLevel::ERROR);
    }

    if ((env_tmp.size() != 1) || (valid_log_levels.find(env_tmp[0]) == valid_log_levels.end())) {
      // undefined log level env, use level ERROR.
      tng::Logger(__FILE__, __LINE__, "WARNING") << \
        "Value of TNG_LOG_LEVEL should be in {0, 1, 2, 3, 4}, but got " << env_tmp;
      return static_cast<int32_t>(tng::LogLevel::ERROR);
    }
    return atoi(env_val);
  }
  return static_cast<int32_t>(tng::LogLevel::ERROR);
}();

ge::char_t *CreateMessage(const ge::char_t *format, va_list arg) {
  if (format == nullptr) {
    return nullptr;
  }

  va_list arg_copy;
  va_copy(arg_copy, arg);
  int len = vsnprintf(nullptr, 0, format, arg_copy);
  va_end(arg_copy);

  if (len < 0) {
    return nullptr;
  }

  auto msg = std::unique_ptr<ge::char_t[]>(new (std::nothrow) ge::char_t[len + 1]);
  if (msg == nullptr) {
    return nullptr;
  }

  auto ret = vsnprintf_s(msg.get(), len + 1, len, format, arg);
  if (ret < 0) {
    return nullptr;
  }

  return msg.release();
}
Status Status::Success() { return {}; }
Status::Status(const Status &other) : status_{nullptr} { *this = other; }
Status &Status::operator=(const Status &other) {
  delete[] status_;
  if (other.status_ == nullptr) {
    status_ = nullptr;
  } else {
    size_t status_len = strlen(other.status_) + 1;
    status_ = new (std::nothrow) ge::char_t[status_len];
    if (status_ != nullptr) {
      auto ret = strcpy_s(status_, status_len, other.status_);
      if (ret != EOK) {
        status_[0] = '\0';
      }
    }
  }
  return *this;
}
Status::Status(Status &&other) noexcept {
  status_ = other.status_;
  other.status_ = nullptr;
}
Status &Status::operator=(Status &&other) noexcept {
  delete[] status_;
  status_ = other.status_;
  other.status_ = nullptr;
  return *this;
}
Status Status::Error(const ge::char_t *message, ...) {
  Status status;
  va_list arg;
  va_start(arg, message);
  status.status_ = CreateMessage(message, arg);
  va_end(arg);
  return status;
}
}  // namespace tng
