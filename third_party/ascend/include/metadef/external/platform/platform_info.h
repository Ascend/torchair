/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef PLATFORM_INFO_H
#define PLATFORM_INFO_H

#include <map>
#include <string>
#include <array>
#include "platform_info_def.h"
#include "platform_infos_def.h"
#include "platform_infos_lite_def.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif

namespace fe {
class PlatformInfoManager {
 public:
  PlatformInfoManager(const PlatformInfoManager &) = delete;
  PlatformInfoManager &operator=(const PlatformInfoManager &) = delete;

  static PlatformInfoManager &Instance();
  static PlatformInfoManager &GeInstance();
  uint32_t InitializePlatformInfo();
  uint32_t Finalize();

  uint32_t GetPlatformInfo(const std::string SoCVersion,
                           PlatformInfo &platform_info,
                           OptionalInfo &opti_compilation_info);

  uint32_t GetPlatformInfoWithOutSocVersion(PlatformInfo &platform_info,
                                            OptionalInfo &opti_compilation_info);

  void SetOptionalCompilationInfo(OptionalInfo &opti_compilation_info);

  uint32_t GetPlatformInfos(const std::string SoCVersion,
                            PlatFormInfos &platform_info,
                            OptionalInfos &opti_compilation_info);

  uint32_t GetPlatformInfoWithOutSocVersion(PlatFormInfos &platform_info,
                                            OptionalInfos &opti_compilation_info);

  void SetOptionalCompilationInfo(OptionalInfos &opti_compilation_info);

  uint32_t UpdatePlatformInfos(PlatFormInfos &platform_infos);

  uint32_t UpdatePlatformInfos(const string &soc_version, PlatFormInfos &platform_infos);

  uint32_t GetPlatformInstanceByDevice(const uint32_t &device_id, PlatFormInfos &platform_infos);

  uint32_t GetRuntimePlatformInfosByDevice(const uint32_t &device_id, PlatFormInfos &platform_infos);

  uint32_t UpdateRuntimePlatformInfosByDevice(const uint32_t &device_id, PlatFormInfos &platform_infos);

  uint32_t InitRuntimePlatformInfos(const std::string &SoCVersion);

  uint32_t GetPlatFormInfosLite(SocVersion soc_version, PlatFormInfosLite &platform_infos_lite);
 private:
  PlatformInfoManager();
  ~PlatformInfoManager();

  uint32_t LoadIniFile(std::string ini_file_real_path);

  void Trim(std::string &str);

  uint32_t LoadConfigFile(std::string real_path);

  std::string RealPath(const std::string &path);

  std::string GetSoFilePath();

  void ParseVersion(std::map<std::string, std::string> &version_map,
                    std::string &soc_version,
                    PlatformInfo &platform_info_temp);

  void ParseSocInfo(std::map<std::string, std::string> &soc_info_map,
                    PlatformInfo &platform_info_temp);

  void ParseCubeOfAICoreSpec(std::map<std::string, std::string> &ai_core_spec_map,
                             PlatformInfo &platform_info_temp);

  void ParseBufferOfAICoreSpec(std::map<std::string, std::string> &ai_core_spec_map,
                               PlatformInfo &platform_info_temp);

  void ParseUBOfAICoreSpec(std::map<std::string, std::string> &ai_core_spec_map,
                           PlatformInfo &platform_info_temp);

  void ParseUnzipOfAICoreSpec(std::map<std::string, std::string> &ai_core_spec_map,
                              PlatformInfo &platform_info_temp);

  void ParseAICoreSpec(std::map<std::string, std::string> &ai_core_spec_map,
                       PlatformInfo &platform_info_temp);

  void ParseBufferOfAICoreMemoryRates(std::map<std::string, std::string> &ai_core_memory_rates_map,
                                      PlatformInfo &platform_info_temp);

  void ParseAICoreMemoryRates(std::map<std::string, std::string> &ai_core_memory_rates_map,
                              PlatformInfo &platform_info_temp);

  void ParseUBOfAICoreMemoryRates(std::map<std::string, std::string> &ai_core_memory_rates_map,
                                  PlatformInfo &platform_info_temp);

  void ParseAICoreintrinsicDtypeMap(std::map<std::string, std::string> &ai_coreintrinsic_dtype_map,
                                    PlatformInfo &platform_info_temp);

  void ParseVectorCoreSpec(std::map<std::string, std::string> &vector_core_spec_map,
                           PlatformInfo &platform_info_temp);

  void ParseVectorCoreMemoryRates(std::map<std::string, std::string> &vector_core_memory_rates_map,
                                  PlatformInfo &platform_info_temp);

  void ParseCPUCache(std::map<std::string, std::string> &CPUCacheMap,
                     PlatformInfo &platform_info_temp);

  void ParseVectorCoreintrinsicDtypeMap(std::map<std::string, std::string> &vector_coreintrinsic_dtype_map,
                                        PlatformInfo &platform_info_temp);

  uint32_t ParsePlatformInfoFromStrToStruct(std::map<std::string, std::map<std::string, std::string>> &content_info_map,
                                            std::string &soc_version,
                                            PlatformInfo &platform_info_temp);

  void ParseAICoreintrinsicDtypeMap(std::map<std::string, std::string> &ai_coreintrinsic_dtype_map,
                                    PlatFormInfos &platform_info_temp);

  void ParseVectorCoreintrinsicDtypeMap(std::map<std::string, std::string> &vector_coreintrinsic_dtype_map,
                                        PlatFormInfos &platform_info_temp);

  void ParseSoftwareSpec(map<string, string> &software_spec_map, PlatformInfo &platform_info_temp);

  void ParsePlatformRes(const std::string &label,
                        std::map<std::string, std::string> &platform_res_map,
                        PlatFormInfos &platform_info_temp);

  uint32_t ParsePlatformInfo(std::map<std::string, std::map<std::string, std::string>> &content_info_map,
                             std::string &soc_version,
                             PlatFormInfos &platform_info_temp);

  uint32_t AssemblePlatformInfoVector(std::map<std::string, std::map<std::string, std::string>> &content_info_map);
  void FillupFixPipeInfo(PlatFormInfos &platform_infos);

  bool init_flag_;
  bool runtime_init_flag_;
  std::map<std::string, PlatformInfo> platform_info_map_;

  OptionalInfo opti_compilation_info_;

  std::map<std::string, PlatFormInfos> platform_infos_map_;

  OptionalInfos opti_compilation_infos_;

  std::map<uint32_t, PlatFormInfos> device_platform_infos_map_;

  PlatFormInfos runtime_platform_infos_;

  std::map<uint32_t, PlatFormInfos> runtime_device_platform_infos_map_;

  std::array<PlatFormInfosLite, TOTAL_SOC_COUNT> platform_infos_lite_vec_;
};
}  // namespace fe

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
