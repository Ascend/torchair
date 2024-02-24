/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
#ifndef INC_GRAPH_GE_CONTEXT_H_
#define INC_GRAPH_GE_CONTEXT_H_

#include <string>
#include <map>
#include <vector>
#include "graph/ge_error_codes.h"

namespace ge {
class GEContext {
 public:
  graphStatus GetOption(const std::string &key, std::string &option);
  const std::string &GetReadableName(const std::string &key);
  bool GetHostExecFlag() const;
  bool GetTrainGraphFlag() const;
  bool IsOverflowDetectionOpen() const;
  bool IsGraphLevelSat() const;
  uint64_t SessionId() const;
  uint32_t DeviceId() const;
  int32_t StreamSyncTimeout() const;
  int32_t EventSyncTimeout() const;
  void Init();
  void SetSessionId(const uint64_t session_id);
  void SetContextId(const uint64_t context_id);
  void SetCtxDeviceId(const uint32_t device_id);
  void SetStreamSyncTimeout(const int32_t timeout);
  void SetEventSyncTimeout(const int32_t timeout);
  graphStatus SetOptionNameMap(const std::string &option_name_map_json);
  void SetMultiBatchShapeIndex(uint32_t graph_id,
      const std::map<int32_t, std::vector<int32_t>> &data_index_and_shape_map);
  const std::map<int32_t, std::vector<int32_t>> GetMultiBatchShapeIndex(uint32_t graph_id);
 private:
  thread_local static uint64_t session_id_;
  thread_local static uint64_t context_id_;
  uint32_t device_id_ = 0U;
  uint64_t trace_id_ = 0U;
  int32_t stream_sync_timeout_ = -1;
  int32_t event_sync_timeout_ = -1;
  std::map<std::string, std::string> option_name_map_;
  // 保存分档挡位信息， 第一层map表的key表示图id，value是保存了第二层map表
  // 第二层map表的key表示算子index，value保存了该Data算子的shape是-1的维度的索引
  std::map<uint32_t, std::map<int32_t, std::vector<int32_t>>> data_index_and_shape_map_;
};  // class GEContext

/// Get context
/// @return
GEContext &GetContext();
}  // namespace ge

#endif  //  INC_GRAPH_GE_CONTEXT_H_
