/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INC_REGISTER_OP_EXTRA_GENTASK_REGISTRY_H
#define INC_REGISTER_OP_EXTRA_GENTASK_REGISTRY_H
#include <string>
#include <functional>
#include <vector>
#include "graph/node.h"
#include "proto/task.pb.h"
#include "external/ge_common/ge_api_types.h"
#include "common/opskernel/ops_kernel_info_types.h"
namespace fe {
using OpExtGenTaskFunc = ge::Status (*)(const ge::Node &node,
                                        ge::RunContext &context, std::vector<domi::TaskDef> &tasks);
class OpExtGenTaskRegistry {
 public:
  OpExtGenTaskRegistry() {};
  ~OpExtGenTaskRegistry() {};
  static OpExtGenTaskRegistry &GetInstance();
  OpExtGenTaskFunc FindRegisterFunc(const std::string &op_type) const;
  void Register(const std::string &op_type, const OpExtGenTaskFunc func);

 private:
    std::unordered_map<std::string, OpExtGenTaskFunc> names_to_register_func_;
};

class OpExtGenTaskRegister {
public:
    OpExtGenTaskRegister(const char *op_type, OpExtGenTaskFunc func) noexcept;
};
}  // namespace fe

#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif

#define REGISTER_NODE_EXT_GENTASK_COUNTER2(type, func, counter)                  \
  static const fe::OpExtGenTaskRegister g_reg_op_ext_gentask_##counter ATTRIBUTE_USED =  \
      fe::OpExtGenTaskRegister(type, func)
#define REGISTER_NODE_EXT_GENTASK_COUNTER(type, func, counter)                    \
  REGISTER_NODE_EXT_GENTASK_COUNTER2(type, func, counter)
#define REGISTER_NODE_EXT_GENTASK(type, func)                                \
  REGISTER_NODE_EXT_GENTASK_COUNTER(type, func, __COUNTER__)
#endif // INC_REGISTER_OP_EXTRA_GENTASK_REGISTRY_H
