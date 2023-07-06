/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef REGISTER_OP_TILING_COMPILE_INFO_MANAGER_H_
#define REGISTER_OP_TILING_COMPILE_INFO_MANAGER_H_

#include <string>
#include <mutex>
#include <unordered_map>

#include "register/op_compile_info_base.h"

namespace optiling {
class CompileInfoManager {
public:
  CompileInfoManager(const CompileInfoManager &) = delete;
  CompileInfoManager &operator=(const CompileInfoManager &) = delete;
  static CompileInfoManager& Instance();
  bool HasCompileInfo(const std::string &key);
  CompileInfoPtr GetCompileInfo(const std::string &key);
  void SetCompileInfo(const std::string &key, CompileInfoPtr compile_info_ptr);

private:
  CompileInfoManager();
  ~CompileInfoManager();
  mutable std::mutex compile_info_mutex_;
  std::unordered_map<std::string, CompileInfoPtr> compile_info_map_;
};

class CompileInfoCache {
public:
  CompileInfoCache(const CompileInfoCache &) = delete;
  CompileInfoCache &operator=(const CompileInfoCache &) = delete;
  static CompileInfoCache& Instance();
  bool HasCompileInfo(const std::string &key);
  void* GetCompileInfo(const std::string &key);
  void SetCompileInfo(const std::string &key, void* value);

private:
  CompileInfoCache();
  ~CompileInfoCache();
  mutable std::mutex compile_info_mutex_;
  std::unordered_map<std::string, void *> compile_info_map_;
};
}  // namespace optiling
#endif  // REGISTER_OP_TILING_COMPILE_INFO_MANAGER_H_
