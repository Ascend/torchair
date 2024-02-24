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

#ifndef PLATFORM_INFOS_DEF_H
#define PLATFORM_INFOS_DEF_H

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "platform_info_def.h"

namespace fe {
enum class LocalMemType {
  L0_A = 0,
  L0_B = 1,
  L0_C = 2,
  L1 = 3,
  L2 = 4,
  UB = 5,
  HBM = 6,
  RESERVED
};
class PlatFormInfosImpl;
using PlatFormInfosImplPtr = std::shared_ptr<PlatFormInfosImpl>;
class PlatFormInfos {
 public:
  bool Init();
  std::map<std::string, std::vector<std::string>> GetAICoreIntrinsicDtype();
  std::map<std::string, std::vector<std::string>> GetVectorCoreIntrinsicDtype();
  bool GetPlatformRes(const std::string &label, const std::string &key, std::string &val);
  bool GetPlatformResWithLock(const std::string &label, const std::string &key, std::string &val);
  bool GetPlatformRes(const std::string &label, std::map<std::string, std::string> &res);
  bool GetPlatformResWithLock(const std::string &label, std::map<std::string, std::string> &res);
  uint32_t GetCoreNum() const;
  uint32_t GetCoreNumWithLock() const;
  void GetLocalMemSize(const LocalMemType &mem_type, uint64_t &size);
  void GetLocalMemBw(const LocalMemType &mem_type, uint64_t &bw_size);

  std::map<std::string, std::vector<std::string>> GetAICoreIntrinsicDtype() const;
  std::map<std::string, std::vector<std::string>> GetVectorCoreIntrinsicDtype() const;
  bool GetPlatformRes(const std::string &label, const std::string &key, std::string &val) const;

  void SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsic_dtypes);
  void SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsic_dtypes);
  void SetPlatformRes(const std::string &label, std::map<std::string, std::string> &res);
  void SetPlatformResWithLock(const std::string &label, std::map<std::string, std::string> &res);
  std::map<std::string, std::vector<std::string>> GetFixPipeDtypeMap();
  void SetFixPipeDtypeMap(const std::map<std::string, std::vector<std::string>> &fixpipe_dtype_map);
  void SetCoreNumByCoreType(const std::string &core_type);
  void SetCoreNum(const uint32_t &core_num);
 private:
  uint32_t core_num_ {0};
  PlatFormInfosImplPtr platform_infos_impl_ {nullptr};
};

class OptionalInfosImpl;
using OptionalInfosImplPtr = std::shared_ptr<OptionalInfosImpl>;
class OptionalInfos {
 public:
  bool Init();
  std::string GetSocVersion();
  std::string GetCoreType();
  uint32_t GetAICoreNum();
  std::string GetL1FusionFlag();
  std::map<std::string, std::vector<std::string>> GetFixPipeDtypeMap();

  std::string GetSocVersion() const;
  std::string GetCoreType() const;
  uint32_t GetAICoreNum() const;
  std::string GetL1FusionFlag() const;
  std::map<std::string, std::vector<std::string>> GetFixPipeDtypeMap() const;
  void SetFixPipeDtypeMap(const std::map<std::string, std::vector<std::string>> &fixpipe_dtype_map);
  void SetSocVersion(std::string soc_version);
  void SetSocVersionWithLock(std::string soc_version);
  void SetCoreType(std::string core_type);
  void SetAICoreNum(uint32_t ai_core_num);
  void SetL1FusionFlag(std::string l1_fusion_flag);
 private:
  OptionalInfosImplPtr optional_infos_impl_ {nullptr};
};
}
#endif
