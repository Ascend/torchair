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
#include "graph/ge_error_codes.h"

namespace ge {
class GEContext {
 public:
  graphStatus GetOption(const std::string &key, std::string &option);
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
 private:
  thread_local static uint64_t session_id_;
  thread_local static uint64_t context_id_;
  uint32_t device_id_ = 0U;
  uint64_t trace_id_ = 0U;
  int32_t stream_sync_timeout_ = -1;
  int32_t event_sync_timeout_ = -1;
};  // class GEContext

/// Get context
/// @return
GEContext &GetContext();
}  // namespace ge

#endif  //  INC_GRAPH_GE_CONTEXT_H_
