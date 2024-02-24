/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_COMMON_CONFIGURATION_H_
#define FUSION_ENGINE_INC_COMMON_CONFIGURATION_H_

#include <map>
#include <set>
#include <mutex>
#include <string>
#include <vector>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "common/aicore_util_types.h"
#include "common/base_config_parser.h"

namespace fe {
using std::string;
enum class CONFIG_PARAM {
  SmallChannel = 0,
  JitCompile,
  VirtualType,
  CompressWeight,
  SparseMatrixWeight,
  ReuseMemory,
  AutoTuneMode,
  BufferOptimize,
  FormatMode,
  ConfigParamBottom
};

enum class CONFIG_STR_PARAM {
  HardwareInfo = 0,
  FusionLicense,
  ConfigStrParamBottom
};

enum class ENV_STR_PARAM {
  AscendOppPath = 0,
  NetworkAnalysis,
  DynamicImplFirst,
  AscendCustomOppPath,
  DumpGeGraph,
  DumpGraphLevel,
  NpuCollectPath,
  MinCompileResourceUsageCtrl,
  EnableAclnn,
  AscendWorkPath,
  EnvStrParamBottom
};

enum class CONFIG_PARSER_PARAM {
  ImplMode = 0,
  CustDtypes,
  ModifyMixlist,
  ConfigParserParamBottom
};

/** @brief Configuration.
* Used to manage all the configuration data within the fusion engine module. */
class Configuration {
 public:
  Configuration(const Configuration &) = delete;
  Configuration &operator=(const Configuration &) = delete;

  /**
   * Get the Singleton Instance by engine name
   */
  static Configuration &Instance(const string &engine_name);

  /**
   * Initialize the content_map and ops_store_info_vector_
   * Read the content from the config file, the save the data into content_map
   * Find the data about the Op Store Info from the content_map
   * and build the ops_store_info_vector_ with them.
   * @return Whether the object has been initialized successfully.
   */
  Status Initialize(const std::map<string, string> &options);

  Status InitializeConfigParser(const std::map<string, string> &options);

  Status InitializeExtend(const std::map<string, string> &options);

  Status Finalize();

  /**
   * Find the FEOpsStoreInfo object with the OpImplType.
   * @param op_impl_type ImplType of OP storeinfo
   * @param op_store_info output value.
   *    if the object has been found, the op_store_info will refer to this object
   * @return Whether the FEOpsStoreInfo object has been found.
   * Status SUCCESS:found, FAILED:not found
   */
  Status GetOpStoreInfoByImplType(OpImplType op_impl_type, FEOpsStoreInfo &op_store_info) const;

  /*
   * to get the OpsStoreInfo out of the current configuration object
   * @return the OpsStoreInfo
   */
  const std::vector<FEOpsStoreInfo> &GetOpsStoreInfo() const;

  void SetOpsStoreInfo(const FEOpsStoreInfo &fe_ops_store_info);

  /*
   *  get small channel
   */
  bool IsEnableSmallChannel() const;

  /*
   *  get jit compile
   */
  JitCompileCfg GetJitCompileCfg() const;

  /*
   *  get virtualization on compute capability
   */
  bool IsEnableVirtualType() const;

  /*
   *  get compress weight
   */
  bool IsEnableCompressWeight() const;

  /*
   *  get compress sparse_matrix
   */
  bool IsEnableSparseMatrixWeight() const;

  const std::map<int32_t, float>& GetCompressRatios() const;

  const float& GetAICoreCompressRatio() const;

  /*
   * to get l1fusion option out of the current configuration object
   * @return true/false
   */
  bool EnableL1Fusion() const;

  bool EnableL2Fusion() const;

  bool GetDuplicationSwitch() const;

  bool IsEnableNetworkAnalysis() const;

  bool IsEnableOpImplStrategy() const;

  bool IsEnableUbFusion() const;

  bool IsEnableAclnn() const;

  bool IsEnableFirstLayerQuantization() const;

  bool IsEnableCustomImplMode() const;
  /*
   * to get switch switch of dump original nodes to fusion node
   * @return true/false
   */
  bool GetDumpOriginalNodesEnable() const;
  /*
   * to get switch switch of mix_l2
   * @return true/false
   */
  bool GetMixL2Enable() const;

  bool IsEnableSuperkernelPlus() const;

  /*
   * to get the soc version out of the current configuration object
   * @return soc version
   */
  const string &GetSocVersion() const;

  /**
   * Get the rootdir from configuration file
   * @return root_path
   */
  string GetRootPath();

  static string GetPrecisionModeStr();

  const string& GetLicenseFusionStr() const;

  AutoTuneMode GetAutoTuneMode() const;

  bool IsEnableReuseMemory() const;
  bool IsConfigDebugListOp(const ge::OpDescPtr &op_des_ptr) const;

  string GetBuiltInFusionConfigFilePath() const;

  Status GetCompatibleBuiltInPath(string &builtin_pass_file_path) const;
  /**
   * Get the fusionpassmgr.graphpasspath from configuration file
   * @return builtin_pass_file_path
   */
  Status GetBuiltinPassFilePath(string &builtin_pass_file_path);

  /**
   * Get the fusionpassmgr.custompasspath from configuration file
   * @return custom_pass_file_path
   */
  Status GetCustomPassFilePath(string &custom_pass_file_path);

  /**
   * Get the fusionrulemgr.graphfilepath from configuration file
   * @return graphfilepath
   */
  Status GetGraphFilePath(string &graph_file_path);

  /**
   * Get the fusionrulemgr.customfilepath from configuration file
   * @return customfilepath
   */
  Status GetCustomFilePath(string &custom_file_path);

  bool IsInLicenseControlMap(const string &key) const;

  string GetFeLibPath() const;

  int32_t GetMemReuseDistThreshold() const;

  const std::unordered_set<string>& GetFp16OpTypeList() const;

  const std::set<string>& GetLicenseFusionDetailInfo() const;

  const std::string& GetDumpGeGraph() const;

  const std::string& GetDumpGraphLevel() const;

  const std::string GetOpDebugConfig() const;

  bool GetOpImplMode(const string &op_name, const string &op_type, string &op_impl_mode) const;

  bool GetCustomizeDtypeByOpType(const string &op_type, OpCustomizeDtype &custom_dtype);

  bool GetCustomizeDtypeByOpName(const string &op_name, OpCustomizeDtype &custom_dtype);

  FormatModeType GetFormatModeCfg() const;

  PrecisionPolicy GetPrecisionPolicy(const std::string &op_type, const PrecisionPolicy &op_kernel_policy);

  const std::map<string, string>& GetBinaryPathMap() const;

  std::string GetAllOpsImplPath() const;

  const std::map<string, string>& GetHardwareInfo() const;

  bool IsDynamicImplFirst() const;

  Status RefreshParameters();

  bool GetMemoryCheckSwitch() const;

  const std::string& GetBinaryConfigFilePath() const;

  const std::string& GetAscendWorkPath() const;

 private:
  explicit Configuration(const string &engine_name);
  ~Configuration();
  bool is_init_;
  string engine_name_;
  string lib_path_;
  std::map<string, string> content_map_;
  std::vector<FEOpsStoreInfo> ops_store_info_vector_;

  bool is_dynamic_impl_first_; // env
  bool enable_network_analysis_;
  bool enable_op_impl_strategy_;
  bool enable_ub_fusion_;
  bool enable_aclnn_;
  string ascend_ops_path_; // env

  std::array<string, static_cast<size_t>(ENV_STR_PARAM::EnvStrParamBottom)> env_str_param_vec_;
  std::array<int64_t, static_cast<size_t>(CONFIG_PARAM::ConfigParamBottom)> config_param_vec_;
  std::array<string, static_cast<size_t>(CONFIG_STR_PARAM::ConfigStrParamBottom)> config_str_param_vec_;
  std::array<std::map<std::string, BaseConfigParserPtr>,
      static_cast<size_t>(CONFIG_PARSER_PARAM::ConfigParserParamBottom)> config_parser_map_vec_;
  std::array<std::mutex,
      static_cast<size_t>(CONFIG_PARSER_PARAM::ConfigParserParamBottom)> config_parser_map_mutex_vec_;

  std::set<string> license_fusion_detail_value_;
  std::map<string, string> hardware_info_map_;

  bool enable_first_layer_quantization_;
  string bin_cfg_file_;
  std::unordered_set<string> fp16_op_type_list_;

  int64_t op_store_priority_count_;
  int32_t mem_reuse_dist_threshold_;

  std::map<int32_t, float> compress_ratios_;
  float ai_core_compress_ratio_;
  std::map<string, string> op_binary_path_map_;

  BaseConfigParserPtr cust_dtypes_parser_;
  BaseConfigParserPtr impl_mode_parser_;
  BaseConfigParserPtr mix_list_parser_;
  BaseConfigParserPtr op_debug_config_parse_;
  std::map<std::string, BaseConfigParserPtr> mix_list_parser_map_;
  mutable std::mutex config_param_mutex_;
  mutable std::mutex mix_list_parser_map_mutex_;
  mutable std::mutex ops_store_info_vector_mutex_;

  void InitParamFromEnv();

  /**
   * Initialize the parameters from options
   * @param options patameters map
   */
  Status InitConfigParamFromOptions(const std::map<string, string> &options);

  Status InitConfigParamFromContext();

  std::string GetConfigStrParamValueFromContext(CONFIG_STR_PARAM config_str_param_enum_type) const;

  int64_t GetConfigParamValueFromContext(CONFIG_PARAM config_param_enum_type) const;

  std::string CombinedParamsKeyFromOptions(CONFIG_PARSER_PARAM config_parser_param_enum_type,
                                           const std::map<string, string> &options) const;

  std::string CombinedParamsKeyFromContext(CONFIG_PARSER_PARAM config_parser_param_enum_type) const;

  BaseConfigParserPtr GetConfigParserFromContext(CONFIG_PARSER_PARAM config_parser_param_enum_type,
                                                 const std::string &combined_params_key);

  Status RefreshImplMode();

  Status RefreshCustDtypes();

  Status RefreshMixList();

  /**
   * Get the real Path of current so lib
   */
  Status InitLibPath();

  /**
   * Get the real Path of ops
   * path of ops is the path of so package + ops_relative_path
   */
  void InitCustomOpStore();
  Status InitAscendOpsPath();

  bool IsPathExistedInOpp(const std::string &path, bool is_full_path) const;
  void ResolveBinaryPath(const std::string &sub_path, const std::string &path_type, const int64_t main_impl_type,
                         bool isOm, const std::string &binaryKey);
  bool AddCustomOpStoreContent(const std::string &full_or_sub_path, const std::string &path_type,
                               const int64_t main_impl_type, const bool is_full_path);
  bool CheckIsValidAbstractPath(const std::string &path) const;

  Status LoadOppConfigFile();

  Status GetCustomOppPathFromOppConfigFile(std::vector<string> &custom_opp_path_vec) const;

  /**
   * Read the content of configuration file(FE_CONFIG_FILE_PATH)
   * Save the data into content_map
   * @return Whether the config file has been loaded successfully.
   */
  Status LoadConfigFile();

  /**
   * Find the OpsStoreInfo from the content_map,
   * then use the data to build up ops_store_info_vector.
   * @return Whether the OpsStoreInfoVector has been built up successfully.
   */
  Status AssembleOpsStoreInfoVector();

  Status AssembleEachOpsStoreInfo(string &op_store_name, std::vector<string> &op_store_vector,
                                  FEOpsStoreInfo &ops_store_info);

  Status VerifyOpStoreVector(std::vector<string> &op_store_vector, const string &op_store_name) const;

  bool IsIgnoreOpStore(const FEOpsStoreInfo &ops_store_info) const;

  Status CheckOpStoreInfo(const FEOpsStoreInfo &op_store_info) const;

  /**
   * Check whether the content_map contain the input key.
   * @param key
   * @return Whether the content_map contain the input key.
   */
  bool ContainKey(const string &key) const;

  /**
   * Get the value from the content_map if the content_map contains the input key.
   * @param key config key
   * @param return_value output value. if the value has been found,
   *                    return_value will refer to this value.
   * @return Whether the vale has been found.
   *         Status SUCCESS:found, FAILED:not found
   */
  Status GetStringValue(const string &key, string &return_value) const;

  /**
   * Find the value from the content_map by the input key,
   * convert the value to bool type and return the bool value
   * return the input default value if the value is not found.
   * @param key
   * @param default_value
   *   This value will be returned if the input key can be found in content_map.
   * @return bool value
   */
  bool GetBoolValue(const string &key, bool default_value) const;

  BufferOptimize GetBufferOptimize() const;

  void SetEnableSmallChannel(const bool &small_channel);

  void InitParametersOfConfigFile();

  void InitISAArchVersion();

  int32_t ParseDataVisitDistThreshold() const;

  void InitMemReuseDistThreshold();

  void InitFp16OpType();

  void InitCompressRatio();

  void ParseHardwareInfo();

  void ParseFusionLicense(const bool &is_config);

  Status ParseVirtualType();

  void InitBinaryConfigFilePath();

  std::vector<string> ParseConfig(const string &key, char pattern) const;

  bool InitFirstLayerQuantization(const std::map<string, string> &options);

  bool GetConfigValueByKey(const std::map<string, string> &options, const string &file_key,
                           const string &cfg_key, string &value, string &file_path) const;
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_CONFIGURATION_H_
