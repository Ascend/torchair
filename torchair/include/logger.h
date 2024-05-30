#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_LOGGER_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_LOGGER_H_

#include <iostream>
#include <sstream>
#include <cstring>

namespace tng {
class Logger : public std::basic_ostringstream<char> {
 public:
  static int32_t kLogLevel;
  size_t prefix_len;
  Logger(const char *f, int line, const char *log_level) {
    *this << "[" << log_level << "] TORCHAIR [" << f << ":" << line << "] ";
    prefix_len = str().length();
  }
  ~Logger() override {
    size_t prev_pos = 0U;
    size_t cur_pos = 0U;
    size_t original_str_len = str().length();

    while ((cur_pos = str().find('\n', cur_pos)) != std::string::npos) {
      std::cerr << str().substr(prev_pos, cur_pos - prev_pos) << std::endl;
      if (cur_pos == original_str_len - 1U) {
        break;
      }
      std::cerr << str().substr(0U, prefix_len);
      cur_pos++;
      prev_pos = cur_pos;
    }
    if (cur_pos != original_str_len - 1U) {
      std::cerr << str().substr(prev_pos, original_str_len - prev_pos) << std::endl;
    }
  }
};

enum class LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  EVENT = 4
};

inline bool LogLevelEnable(const int32_t current, const int32_t configured) {
  if (current == static_cast<int32_t>(tng::LogLevel::ERROR)) {
    return true;
  }
  if (current == static_cast<int32_t>(tng::LogLevel::EVENT)) {
    return configured == static_cast<int32_t>(tng::LogLevel::EVENT);
  }
  return (current >= configured);
}
}  // namespace tng

#define TNG_LOG(L) \
  if (tng::LogLevelEnable(static_cast<int32_t>(tng::LogLevel::L), static_cast<int32_t>(tng::Logger::kLogLevel))) \
    tng::Logger(__FILE__, __LINE__, (#L))

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_LOGGER_H_