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

#ifndef GE_COMMON_GE_PLUGIN_MANAGER_H_
#define GE_COMMON_GE_PLUGIN_MANAGER_H_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "common/ge_common/ge_inner_error_codes.h"
#include "common/ge_common/debug/ge_log.h"
#include "mmpa/mmpa_api.h"

namespace ge {
class DNNEngine;
class PluginManager {
 public:
  PluginManager() = default;

  ~PluginManager();

  static void SplitPath(const std::string &mutil_path, std::vector<std::string> &path_vec, const char sep = ':');

  static Status GetOppPath(std::string &opp_path);

  static bool IsNewOppPathStruct(const std::string &opp_path);

  static Status GetOppPluginVendors(const std::string &vendors_config, std::vector<std::string> &vendors);

  static Status ReversePathString(std::string &path_str);

  static void GetPluginPathFromCustomOppPath(const std::string &sub_path, std::string &plugin_path);

  static Status GetOppPluginPathOld(const std::string &opp_path,
                                    const std::string &path_fmt,
                                    std::string &plugin_path,
                                    const std::string &path_fmt_custom = "");

  static Status GetOppPluginPathNew(const std::string &opp_path,
                                    const std::string &path_fmt,
                                    std::string &plugin_path,
                                    const std::string &old_custom_path,
                                    const std::string &path_fmt_custom = "");

  static Status GetOpsProtoPath(std::string &opsproto_path);

  static Status GetCustomOpPath(const std::string &fmk_type, std::string &customop_path);

  static Status GetCustomCaffeProtoPath(std::string &customcaffe_path);

  static Status GetOpTilingPath(std::string &op_tiling_path);

  static Status GetOpTilingForwardOrderPath(std::string &op_tiling_path);

  static Status GetConstantFoldingOpsPath(const std::string &path_base, std::string &constant_folding_ops_path);

  Status LoadSoWithFlags(const std::string &path, const int32_t flags,
      const std::vector<std::string> &func_check_list = std::vector<std::string>());

  Status LoadSo(const std::string &path, const std::vector<std::string> &func_check_list = std::vector<std::string>());

  Status Load(const std::string &path, const std::vector<std::string> &func_check_list = std::vector<std::string>());

  Status LoadWithFlags(const std::string &path, const int32_t flags,
      const std::vector<std::string> &func_check_list = std::vector<std::string>());

  static void GetOppSupportedOsAndCpuType(
      std::unordered_map<std::string, std::unordered_set<std::string>> &opp_supported_os_cpu,
      std::string opp_path = "", std::string os_name = "", uint32_t layer = 0U);

  static void GetCurEnvPackageOsAndCpuType(std::string &host_env_os, std::string &host_env_cpu);

  static bool GetVersionFromPath(const std::string &file_path, std::string &version);

  static void GetFileListWithSuffix(const std::string &path, const std::string &so_suff,
                                    std::vector<std::string> &file_list);

  static bool IsVendorVersionValid(const std::string &opp_version, const std::string &compiler_version);

  static bool IsVendorVersionValid(const std::string &vendor_path);

  static void GetPackageSoPath(std::vector<std::string> &vendors);

  static bool GetVersionFromPathWithName(const std::string &file_path, std::string &version,
                                         const std::string version_name);

  template <typename R, typename... Types>
  Status GetAllFunctions(const std::string &func_name, std::map<std::string, std::function<R(Types... args)>> &funcs) {
    for (const auto &handle : handles_) {
      const auto real_fn = reinterpret_cast<R(*)(Types...)>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to get function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return GE_PLGMGR_FUNC_NOT_EXIST;
      } else {
        funcs[handle.first] = real_fn;
      }
    }
    return SUCCESS;
  }

  template <typename... Types>
  Status InvokeAll(const std::string &func_name, const Types... args) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      const auto real_fn = reinterpret_cast<void (*)(Types...)>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return GE_PLGMGR_INVOKE_FAILED;
      } else {
        real_fn(args...);
      }
    }
    return SUCCESS;
  }

  template <typename T>
  Status InvokeAll(const std::string &func_name, const T arg) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      const auto real_fn = reinterpret_cast<void (*)(T)>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        const char_t *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return GE_PLGMGR_INVOKE_FAILED;
      }
      typename std::remove_reference<T>::type arg_temp;
      real_fn(arg_temp);

      if (std::is_same<typename std::remove_reference<T>::type,
                       std::map<std::string, std::shared_ptr<DNNEngine>>>::value) {
        for (const auto &val : arg_temp) {
          if (arg.find(val.first) != arg.end()) {
            GELOGW("FuncName %s in so %s find the same key: %s, will replace it", func_name.c_str(),
                   handle.first.c_str(), val.first.c_str());
            arg[val.first] = val.second;
          }
        }
      }
      arg.insert(arg_temp.begin(), arg_temp.end());
    }
    return SUCCESS;
  }

  template <typename... Args>
  void OptionalInvokeAll(const std::string &func_name, const Args... args) const {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      const auto real_fn = reinterpret_cast<void (*)(Args...)>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        GELOGI("func %s not exist in so %s", handle.first.c_str(), func_name.c_str());
        continue;
      } else {
        GELOGI("func %s exists in so %s", handle.first.c_str(), func_name.c_str());
        real_fn(args...);
      }
    }
  }

  template <typename T1, typename T2>
  Status InvokeAll(const std::string &func_name, const T1 arg) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      const auto real_fn = reinterpret_cast<T2(*)(T1)>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        const char_t *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return GE_PLGMGR_INVOKE_FAILED;
      } else {
        const T2 res = real_fn(arg);
        if (res != SUCCESS) {
          return FAILED;
        }
      }
    }
    return SUCCESS;
  }

  template <typename T>
  Status InvokeAll(const std::string &func_name) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      const auto real_fn = reinterpret_cast<T(*)()>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        const char_t *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return GE_PLGMGR_INVOKE_FAILED;
      } else {
        const T res = real_fn();
        if (res != SUCCESS) {
          return FAILED;
        }
      }
    }
    return SUCCESS;
  }

 private:
  void ClearHandles_() noexcept;
  Status ValidateSo(const std::string &file_path, const int64_t size_of_loaded_so, int64_t &file_size) const;
  static bool ParseVersion(std::string &line, std::string &version, const std::string version_name);
  static bool GetRequiredOppAbiVersion(std::vector<std::pair<uint32_t, uint32_t>> &required_opp_abi_version);
  static bool GetEffectiveVersion(const std::string &opp_version, uint32_t &effective_version);
  static bool CheckOppAndCompilerVersions(const std::string &opp_version, const std::string &compiler_version,
                                          const std::vector<std::pair<uint32_t, uint32_t>> &required_version);
  static void GetOppAndCompilerVersion(const std::string &vendor_path, std::string &opp_version,
                                       std::string &compiler_version);
  std::vector<std::string> so_list_;
  std::map<std::string, void *> handles_;
};

inline std::string GetModelPath() {
  mmDlInfo dl_info;
  if ((mmDladdr(reinterpret_cast<void *>(&GetModelPath), &dl_info) != EN_OK) || (dl_info.dli_fname == nullptr)) {
    GELOGW("Failed to read the shared library file path! errmsg:%s", mmDlerror());
    return std::string();
  }

  if (strlen(dl_info.dli_fname) >= MMPA_MAX_PATH) {
    GELOGW("The shared library file path is too long!");
    return std::string();
  }

  char_t path[MMPA_MAX_PATH] = {};
  if (mmRealPath(dl_info.dli_fname, &path[0], MMPA_MAX_PATH) != EN_OK) {
    constexpr size_t max_error_strlen = 128U;
    char_t err_buf[max_error_strlen + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], max_error_strlen);
    GELOGW("Failed to get realpath of %s, errmsg:%s", dl_info.dli_fname, err_msg);
    return std::string();
  }

  std::string so_path = path;
  so_path = so_path.substr(0U, so_path.rfind('/') + 1U);
  return so_path;
}
}  // namespace ge

#endif  // GE_COMMON_GE_PLUGIN_MANAGER_H_
