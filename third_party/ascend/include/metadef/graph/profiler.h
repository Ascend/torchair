/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef METADEF_CXX_PROFILER_H
#define METADEF_CXX_PROFILER_H
#include <memory>
#include <array>
#include <ostream>
#include <chrono>
#include <atomic>
#include "external/graph/types.h"

namespace ge {
namespace profiling {
constexpr size_t kMaxStrLen = 256UL;
constexpr int64_t kMaxStrIndex = 1024 * 1024;
constexpr size_t kMaxRecordNum = 10UL * 1024UL * 1024UL;
enum class EventType {
  kEventStart,
  kEventEnd,
  kEventTimestamp,
  kEventTypeEnd
};
struct ProfilingRecord {
  int64_t element;
  int64_t thread;
  int64_t event;
  EventType et;
  std::chrono::time_point<std::chrono::system_clock> timestamp;
};

struct StrHash {
    char_t str[kMaxStrLen];
    uint64_t hash;
};

class Profiler {
 public:
  static std::unique_ptr<Profiler> Create();
  void UpdateHashByIndex(const int64_t index, const uint64_t hash);
  void RegisterString(const int64_t index, const std::string &str);
  void RegisterStringHash(const int64_t index, const uint64_t hash, const std::string &str);
  void Record(const int64_t element, const int64_t thread, const int64_t event, const EventType et,
              const std::chrono::time_point<std::chrono::system_clock> time_point);
  void RecordCurrentThread(const int64_t element, const int64_t event, const EventType et);
  void RecordCurrentThread(const int64_t element, const int64_t event, const EventType et,
                           const std::chrono::time_point<std::chrono::system_clock> time_point);

  void Reset();
  void Dump(std::ostream &out_stream) const;

  size_t GetRecordNum() const noexcept;
  const ProfilingRecord *GetRecords() const;

  using ConstStringHashesPointer = StrHash const(*);
  using StringHashesPointer = StrHash (*);
  ConstStringHashesPointer GetStringHashes() const;
  StringHashesPointer GetStringHashes() ;

  ~Profiler();
  Profiler();

 private:
  void DumpByIndex(const int64_t index, std::ostream &out_stream) const;

 private:
  std::atomic<size_t> record_size_;
  std::array<ProfilingRecord, kMaxRecordNum> records_;
  StrHash indexes_to_str_hashes_[kMaxStrIndex];
};
}
}
#endif  // METADEF_CXX_PROFILER_H
