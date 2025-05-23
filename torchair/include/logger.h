#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_LOGGER_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_LOGGER_H_

#include <sys/syscall.h>
#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <iomanip>
#include <chrono>
#include <ctime>

namespace tng {
constexpr int32_t time_width = 3;

inline std::string GetCurrentTimeStamp() {
  auto now = std::chrono::high_resolution_clock::now();
  auto time_now = std::chrono::system_clock::to_time_t(now);
  std::tm* local_time = std::localtime(&time_now);

  auto duration = now.time_since_epoch();
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() % 1000;

  std::ostringstream ss;
  ss << std::put_time(local_time, "%Y-%m-%d-%H:%M:%S");
  ss << "." << std::setfill('0') << std::setw(time_width) << milliseconds;
  ss << "." << std::setfill('0') << std::setw(time_width) << microseconds;
  return ss.str();
}

inline std::string GetProcessIdAndName() {
  static std::string process_id_and_name = []() {
    std::stringstream ss;
    std::string process_name;
    pid_t pid = getpid();
    ss << pid;
    std::stringstream path_ss;
    path_ss << "/proc/" << pid << "/cmdline";
    std::ifstream file(path_ss.str());
    if (file.is_open()) {
      std::getline(file, process_name, '\0');
      file.close();
      size_t pos = process_name.find_last_of('/');
      if (pos != std::string::npos) {
        ss << "," << process_name.substr(pos + 1);
      } else {
        ss << "," << process_name;
      }
    } else {
      ss << ",unknown";
    }

    return ss.str();
  }();
  return process_id_and_name;
}

class Logger : public std::basic_ostringstream<char> {
 public:
  static int32_t kLogLevel;
  size_t prefix_len;
  Logger(const char *f, int line, const char *log_level) {
    *this << "[" << log_level << "] TORCHAIR(" << GetProcessIdAndName() << "):" \
          << GetCurrentTimeStamp() << " [" << f << ":" << line << "]" << syscall(SYS_gettid) << " ";
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

#define TNG_LOG(L) \
  if (tng::LogLevelEnable(static_cast<int32_t>(tng::LogLevel::L), static_cast<int32_t>(tng::Logger::kLogLevel))) \
    tng::Logger(__FILE__, __LINE__, (#L))

inline uint64_t GetTimestampForEventLog() {
  if (tng::Logger::kLogLevel != static_cast<int32_t>(tng::LogLevel::EVENT)) {
    return 0;
  }
  struct timeval tv{};
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    TNG_LOG(ERROR) << "gettimeofday may failed, ret=" << ret;
  }
  auto total_use_time = tv.tv_usec + tv.tv_sec * 1000000; // 1000000: seconds to microseconds
  return static_cast<uint64_t>(total_use_time);
}

inline bool LogLevelEnable(const tng::LogLevel level) {
    return tng::LogLevelEnable(static_cast<int32_t>(level), static_cast<int32_t>(tng::Logger::kLogLevel));
}
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_LOGGER_H_
