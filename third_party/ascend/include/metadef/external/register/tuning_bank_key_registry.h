/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef __INC_REGISTER_TUNING_BANK_KEY_REGISTRY_HEADER__
#define __INC_REGISTER_TUNING_BANK_KEY_REGISTRY_HEADER__
#include <memory>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "graph/ascend_string.h"
#include "common/ge_common/debug/ge_log.h"
#include "external/register/register_types.h"
#include "exe_graph/runtime/tiling_context.h"

#define REGISTER_OP_BANK_KEY_CONVERT_FUN(op, opfunc)                                                                   \
  REGISTER_OP_BANK_KEY_CONVERT_FUN_UNIQ_HELPER(op, (opfunc), __COUNTER__)

#define REGISTER_OP_BANK_KEY_CONVERT_FUN_UNIQ_HELPER(optype, opfunc, counter)                                          \
  REGISTER_OP_BANK_KEY_UNIQ(optype, (opfunc), counter)

#define REGISTER_OP_BANK_KEY_UNIQ(optype, opfunc, counter)                                                             \
  static tuningtiling::OpBankKeyFuncRegistry g_##optype##BankKeyRegistryInterf##counter(#optype, (opfunc))

#define REGISTER_OP_BANK_KEY_PARSE_FUN(op, parse_func, load_func)                                                      \
  REGISTER_OP_BANK_KEY_PARSE_FUN_UNIQ_HELPER(op, (parse_func), (load_func), __COUNTER__)

#define REGISTER_OP_BANK_KEY_PARSE_FUN_UNIQ_HELPER(optype, parse_func, load_func, counter)                             \
  REGISTER_OP_BANK_KEY_PARSE_UNIQ(optype, (parse_func), (load_func), counter)

#define REGISTER_OP_BANK_KEY_PARSE_UNIQ(optype, parse_func, load_func, counter)                                        \
  static tuningtiling::OpBankKeyFuncRegistry g_##optype##BankParseInterf##counter(#optype, (parse_func), (load_func))

#define TUNING_TILING_MAKE_SHARED(exec_expr0, exec_expr1)                                                              \
  do {                                                                                                                 \
    try {                                                                                                              \
      exec_expr0;                                                                                                      \
    } catch (...) {                                                                                                    \
      GELOGW("Make shared failed");                                                                                    \
      exec_expr1;                                                                                                      \
    }                                                                                                                  \
  } while (0)

#define DECLARE_STRUCT_RELATE_WITH_OP(op, bank_key, ...)                                                               \
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(bank_key, __VA_ARGS__);                                                           \
  bool ParseFunc##op##bank_key(const std::shared_ptr<void> &in_args, size_t len, nlohmann::json &bank_key_json) {      \
    if (sizeof(bank_key) != len || in_args == nullptr) {                                                               \
      return false;                                                                                                    \
    }                                                                                                                  \
    bank_key_json = *(std::static_pointer_cast<bank_key>(in_args));                                                    \
    return true;                                                                                                       \
  }                                                                                                                    \
  bool LoadFunc##op##bank_key(std::shared_ptr<void> &in_args, size_t &len, const nlohmann::json &bank_key_json) {      \
    len = sizeof(bank_key);                                                                                            \
    TUNING_TILING_MAKE_SHARED(in_args = std::make_shared<bank_key>(), return false);                                   \
    auto op_ky = std::static_pointer_cast<bank_key>(in_args);                                                          \
    *op_ky = bank_key_json.get<bank_key>();                                                                            \
    return true;                                                                                                       \
  }                                                                                                                    \
  REGISTER_OP_BANK_KEY_PARSE_FUN(op, ParseFunc##op##bank_key, LoadFunc##op##bank_key);


namespace tuningtiling {
using OpBankKeyConvertFun = std::function<bool(const gert::TilingContext *, std::shared_ptr<void> &, size_t &)>;
using OpBankParseFun = std::function<bool(const std::shared_ptr<void> &, size_t, nlohmann::json &)>;
using OpBankLoadFun = std::function<bool(std::shared_ptr<void> &, size_t &, const nlohmann::json &)>;
class FMK_FUNC_HOST_VISIBILITY OpBankKeyFuncInfo {
public:
  explicit OpBankKeyFuncInfo(const ge::AscendString &optype);
  OpBankKeyFuncInfo() = default;
  ~OpBankKeyFuncInfo() = default;
  void SetOpConvertFunc(const OpBankKeyConvertFun &convert_func);
  void SetOpParseFunc(const OpBankParseFun &parse_func);
  void SetOpLoadFunc(const OpBankLoadFun &load_func);
  const OpBankKeyConvertFun& GetBankKeyConvertFunc() const;
  const OpBankParseFun& GetBankKeyParseFunc() const;
  const OpBankLoadFun& GetBankKeyLoadFunc() const;
  const ge::AscendString& GetOpType() const {
    return optype_;
  }
private:
  ge::AscendString optype_;
  OpBankKeyConvertFun convert_func_;
  OpBankParseFun parse_func_;
  OpBankLoadFun load_func_;
};

class FMK_FUNC_HOST_VISIBILITY OpBankKeyFuncRegistry {
public:
  OpBankKeyFuncRegistry(const ge::AscendString &optype, const OpBankKeyConvertFun &convert_func);
  OpBankKeyFuncRegistry(const ge::AscendString &optype, const OpBankParseFun &parse_func,
    const OpBankLoadFun &load_func);
  ~OpBankKeyFuncRegistry() = default;
  static std::unordered_map<ge::AscendString, OpBankKeyFuncInfo> &RegisteredOpFuncInfo();
};
}  // namespace tuningtiling
#endif
