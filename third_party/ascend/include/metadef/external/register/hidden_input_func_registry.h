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
