#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_LOGGER_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_LOGGER_H_

#include <iostream>
#include <sstream>

namespace tng {
class Logger : public std::basic_ostringstream<char> {
 public:
  Logger(const char *f, int line, const char *log_level) {
    *this << "[" << log_level << "] TORCHAIR [" << f << ":" << line << "] ";
  }
  ~Logger() override { std::cerr << str() << std::endl; }
};

enum class LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, FATAL = 4 };
}  // namespace tng
const static int32_t kLogLevel =
  std::getenv("TNG_LOG_LEVEL") ? atoi(std::getenv("TNG_LOG_LEVEL")) : static_cast<int32_t>(tng::LogLevel::ERROR);

#define TNG_LOG(L) \
  if (static_cast<int32_t>(tng::LogLevel::L) >= kLogLevel) tng::Logger(__FILE__, __LINE__, (#L))

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_LOGGER_H_