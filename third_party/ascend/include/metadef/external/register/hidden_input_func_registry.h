/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_REGISTER_HIDDEN_INPUT_FUNC_REGISTRY_H_
#define INC_EXTERNAL_REGISTER_HIDDEN_INPUT_FUNC_REGISTRY_H_

#include <functional>
#include <string>
#include "graph/op_desc.h"
namespace ge {
enum class HiddenInputType : uint32_t { HCOM };

using GetHiddenAddr = ge::graphStatus (*)(const ge::OpDescPtr &op_desc, void *&addr);
class HiddenInputFuncRegistry {
 public:
  static HiddenInputFuncRegistry &GetInstance();
  GetHiddenAddr FindHiddenInputFunc(const HiddenInputType input_type);
  void Register(const HiddenInputType input_type, const GetHiddenAddr func);

 private:
  std::map<HiddenInputType, GetHiddenAddr> type_to_funcs_;
};

class HiddenInputFuncRegister {
 public:
  HiddenInputFuncRegister(const HiddenInputType input_type, const GetHiddenAddr func);
};
}  // namespace ge

#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif
#define REG_HIDDEN_INPUT_FUNC(type, func) REG_HIDDEN_INPUT_FUNC_UNIQ_HELPER(type, func, __COUNTER__)
#define REG_HIDDEN_INPUT_FUNC_UNIQ_HELPER(type, func, counter) REG_HIDDEN_INPUT_FUNC_UNIQ(type, func, counter)
#define REG_HIDDEN_INPUT_FUNC_UNIQ(type, func, counter)                                                                \
  static ::ge::HiddenInputFuncRegister register_hidden_func_##counter ATTRIBUTE_USED =                                 \
      ge::HiddenInputFuncRegister(type, func)

#endif  // INC_EXTERNAL_REGISTER_HIDDEN_INPUT_FUNC_REGISTRY_H_
