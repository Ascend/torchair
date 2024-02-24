/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef METADEF_CXX_INC_EXTERNAL_HCOM_HCOM_TOPO_INFO_H_
#define METADEF_CXX_INC_EXTERNAL_HCOM_HCOM_TOPO_INFO_H_

#include <unordered_map>
#include "ge_common/ge_api_types.h"

namespace ge {
class HcomTopoInfo {
 public:
  struct TopoInfo {
    int64_t rank_size;
    void *notify_handle;
  };
  static HcomTopoInfo &Instance();
  bool TopoInfoHasBeenSet(const char_t *group);
  bool TryGetGroupTopoInfo(const char_t *group, TopoInfo &info);
  Status SetGroupTopoInfo(const char_t *group, const TopoInfo &info);
  Status GetGroupRankSize(const char_t *group, int64_t &rank_size);
  Status GetGroupNotifyHandle(const char_t *group, void *&notify_handle);
  void UnsetGroupTopoInfo(const char_t *group) {
    (void) rank_info_.erase(group);
  };
 private:
  HcomTopoInfo() = default;
  ~HcomTopoInfo() = default;
  std::unordered_map<std::string, TopoInfo> rank_info_;
};
}

#endif // METADEF_CXX_INC_EXTERNAL_HCOM_HCOM_TOPO_INFO_H_
