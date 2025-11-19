#include "logger.h"
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

namespace tng {
std::mutex Logger::g_log_mutex;
std::string Logger::g_torchairDebugLogPath = "";

void SetDebugLogPath(const std::string& path) {
  Logger::SetDebugLogPath(path);
}

void Logger::SetDebugLogPath(const std::string& path) {
  g_torchairDebugLogPath = path;
}

const std::string& Logger::GetDebugLogPath() {
  return g_torchairDebugLogPath;
}

void Logger::WriteLine(const std::string& line, const std::string& log_path) const {
  std::cerr << line << std::endl;
  if (!log_path.empty()) {
    std::ofstream f(log_path, std::ios::app);
    if (f.is_open()) {
      f << line << std::endl;
      f.close(); 
    }
  }
}

Logger::~Logger() {
  const std::string& s = str();
  std::lock_guard<std::mutex> lk(Logger::g_log_mutex); 
  const std::string& log_path = GetDebugLogPath();
  size_t prev_pos = 0U;
  size_t cur_pos = 0U;
  const size_t original_str_len = s.length();

  while ((cur_pos = s.find('\n', cur_pos)) != std::string::npos) {
    const std::string line = s.substr(prev_pos, cur_pos - prev_pos);
    WriteLine(line, log_path);
    if (cur_pos == original_str_len - 1U) {
      break;
    }
    std::cerr << s.substr(0U, prefix_len);
    cur_pos++;
    prev_pos = cur_pos;
  }
  if (cur_pos != original_str_len - 1U) {
    const std::string line = s.substr(prev_pos, original_str_len - prev_pos);
    WriteLine(line, log_path);
  }
} 
} 
