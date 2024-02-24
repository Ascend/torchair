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

#ifndef COMMON_UTIL_TRACE_MANAGER_TRACE_MANAGER_H_
#define COMMON_UTIL_TRACE_MANAGER_TRACE_MANAGER_H_

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <condition_variable>
#include "common/ge_common/util.h"

namespace ge {
#define TRACE_GEN_RECORD(owner, action, graph_name, node_name, node_data, tensor_index, tensor_data, content)      \
  do {                                                                                                             \
    if (TraceManager::GetInstance().IsTraceEnabled()) {                                                            \
      if (TraceManager::GetTraceHeader().size() == 0) {                                                            \
        GELOGD("[Check][Param] owner and stage have not been set");                                                \
      } else {                                                                                                     \
        std::stringstream ss;                                                                                      \
        ss << owner << "," << action << "," << graph_name << "," << node_name << "," << node_data << ","           \
           << tensor_index << "," << tensor_data << "," << content;                                                \
        TraceManager::GetInstance().AddTrace(ss.str());                                                            \
      }                                                                                                            \
    }                                                                                                              \
  } while (false)

using char_t = char;

constexpr uint64_t kTraceSaveTriggerNum = 5000U;

enum class ReadyPart { A, B, None };

class TraceManager {
 public:
  static TraceManager &GetInstance();

  void AddTrace(std::string &&trace_info);

  bool IsTraceEnabled() const {
    return enabled_;
  }
  void SetTraceOwner(const std::string &owner, const std::string &stage, const std::string &graph_name);
  void ClearTraceOwner();
  static inline const std::string &GetTraceHeader() {
    return trace_header_;
  }
  static inline const std::string &GetOutGraphName() {
    return graph_name_;
  }

 private:
  TraceManager();
  ~TraceManager();
  TraceManager(const TraceManager &) = delete;
  TraceManager(TraceManager &&) = delete;
  TraceManager &operator=(const TraceManager &) = delete;
  TraceManager &operator=(TraceManager &&) = delete;
  Status Initialize(const char_t *file_save_path);
  void Finalize();

  std::string NextFileName();
  void SaveTraceBufferToFile(const ReadyPart ready_part);
  void SaveBufferToFileThreadFunc();

  static thread_local std::string trace_header_;
  static thread_local std::string graph_name_;

  std::atomic<bool> enabled_{false};
  std::vector<std::string> trace_array_;
  std::atomic<uint64_t> trace_index_{0};
  std::atomic<uint64_t> total_saved_nums_{0};
  std::atomic<uint64_t> part1_ready_nums_{0};
  std::atomic<uint64_t> part2_ready_nums_{0};
  std::string trace_save_file_path_;
  std::string current_saving_file_name_;
  uint64_t current_file_saved_nums_ = 0;
  ReadyPart ready_part_ = ReadyPart::None;

  std::mutex mu_;
  std::thread save_thread_;
  std::atomic<bool> stopped_{false};
  std::condition_variable data_ready_var_;
};

class TraceOwnerGuard {
 public:
  TraceOwnerGuard(const std::string &owner, const std::string &stage, const std::string &graph_name) {
    TraceManager::GetInstance().SetTraceOwner(owner, stage, graph_name);
  }
  ~TraceOwnerGuard() {
    TraceManager::GetInstance().ClearTraceOwner();
  }
  TraceOwnerGuard(const TraceOwnerGuard &) = delete;
  TraceOwnerGuard(TraceOwnerGuard &&) = delete;
  TraceOwnerGuard &operator=(const TraceOwnerGuard &) = delete;
  TraceOwnerGuard &operator=(TraceOwnerGuard &&) = delete;
};

#define TRACE TraceManager::GetInstance()
#define TRACE_HEADER TraceManager::GetTraceHeader()
}  // namespace ge
#endif  // COMMON_UTIL_TRACE_MANAGER_TRACE_MANAGER_H_
