/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef INC_REGISTER_TIK2_OP_CHECK_H_
#define INC_REGISTER_TIK2_OP_CHECK_H_

#include <map>

#include "graph/ascend_string.h"
#include "graph/operator.h"

namespace optiling {
#define FUNC_CHECK_SUPPORTED "check_supported"
#define FUNC_OP_SELECT_FORMAT "op_select_format"
#define FUNC_GET_OP_SUPPORT_INFO "get_op_support_info"
#define FUNC_GET_SPECIFIC_INFO "get_op_specific_info"

using OP_CHECK_FUNC = int32_t (*)(const ge::Operator &op, ge::AscendString &result);

using PARAM_GENERALIZE_FUNC = int32_t (*)(const ge::Operator &op, const ge::AscendString &generalize_config,
                                      ge::AscendString &generalized_op_params);

struct ReplayFuncParam {
  int32_t block_dim = 0;
  const char *tiling_data = nullptr;
  const char *kernel_name = nullptr;
  const char *entry_file = nullptr;
  int32_t gentype = 0;
  const char *output_kernel_file = nullptr;
  char **objptr = nullptr;
  int32_t task_ration = 0;
  int32_t tiling_key = 0;
};

using REPLAY_FUNC = int32_t (*)(ReplayFuncParam &param, const int32_t core_type);

class OpCheckFuncRegistry {
public:
  static void RegisterOpCapability(const ge::AscendString &check_type, const ge::AscendString &op_type,
                                   OP_CHECK_FUNC func);

  static OP_CHECK_FUNC GetOpCapability(const ge::AscendString &check_type, const ge::AscendString &op_type);

  static PARAM_GENERALIZE_FUNC GetParamGeneralize(const ge::AscendString &op_type);

  static void RegisterParamGeneralize(const ge::AscendString &op_type, PARAM_GENERALIZE_FUNC func);

  static void RegisterReplay(const ge::AscendString &op_type, const ge::AscendString &soc_version, REPLAY_FUNC func);
  static REPLAY_FUNC GetReplay(const ge::AscendString &op_type, const ge::AscendString &soc_version);

private:
  static std::map<ge::AscendString, std::map<ge::AscendString, OP_CHECK_FUNC>> check_op_capability_instance_;
  static std::map<ge::AscendString, PARAM_GENERALIZE_FUNC> param_generalize_instance_;
  static std::map<ge::AscendString, std::map<ge::AscendString, REPLAY_FUNC>> replay_instance_;
};

class OpCheckFuncHelper {
public:
  OpCheckFuncHelper(const ge::AscendString &check_type, const ge::AscendString &op_type, OP_CHECK_FUNC func);

  OpCheckFuncHelper(const ge::AscendString &op_type, PARAM_GENERALIZE_FUNC func);
};

class ReplayFuncHelper {
public:
  ReplayFuncHelper(const ge::AscendString &op_type, const ge::AscendString &soc_version, REPLAY_FUNC func);
};

#define REG_CHECK_SUPPORT(op_type, func)                                                                               \
  static OpCheckFuncHelper op_check_registry_##op_type##_check_supported(FUNC_CHECK_SUPPORTED, #op_type, func)
#define REG_OP_SELECT_FORMAT(op_type, func)                                                                            \
  static OpCheckFuncHelper op_check_registry_##op_type##_op_select_format(FUNC_OP_SELECT_FORMAT, #op_type, func)
#define REG_OP_SUPPORT_INFO(op_type, func)                                                                             \
  static OpCheckFuncHelper op_check_registry_##op_type##_get_op_support_info(FUNC_GET_OP_SUPPORT_INFO, #op_type, func)
#define REG_OP_SPEC_INFO(op_type, func)                                                                                \
  static OpCheckFuncHelper op_check_registry_##op_type##_get_specific_info(FUNC_GET_SPECIFIC_INFO, #op_type, func)

#define REG_OP_PARAM_GENERALIZE(op_type, generalize_func)                                                              \
  static OpCheckFuncHelper op_check_generalize_registry_##op_type(#op_type, generalize_func)

#define REG_REPLAY_FUNC(op_type, soc_version, func)                                                                    \
  static ReplayFuncHelper op_replay_registry_##op_type_##soc_version(#op_type, #soc_version, func)
}  // end of namespace optiling
#endif  // INC_REGISTER_TIK2_OP_CHECK_H_
