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

#ifndef INC_GRAPH_OPSPROTO_MANAGER_H_
#define INC_GRAPH_OPSPROTO_MANAGER_H_

#include <map>
#include <string>
#include <vector>
#include <mutex>

namespace ge {
class OpsProtoManager {
 public:
  static OpsProtoManager *Instance();

  bool Initialize(const std::map<std::string, std::string> &options);
  void Finalize();

 private:
  void LoadOpsProtoPluginSo(const std::string &path);

  std::string pluginPath_;
  std::vector<void *> handles_;
  bool is_init_ = false;
  std::mutex mutex_;
};
}  // namespace ge

#endif  // INC_GRAPH_OPSPROTO_MANAGER_H_
