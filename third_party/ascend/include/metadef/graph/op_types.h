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

#ifndef INC_GRAPH_OP_TYPES_H_
#define INC_GRAPH_OP_TYPES_H_

#include <set>
#include <string>

#include "graph/types.h"

namespace ge {
class GE_FUNC_VISIBILITY OpTypeContainer {
 public:
  static OpTypeContainer &Instance() {
    static OpTypeContainer instance;
    return instance;
  }
  ~OpTypeContainer() = default;

  void Register(const std::string &op_type) { static_cast<void>(op_type_list_.insert(op_type)); }

  bool IsExisting(const std::string &op_type) {
    return op_type_list_.find(op_type) != op_type_list_.end();
  }

 protected:
  OpTypeContainer() {}

 private:
  std::set<std::string> op_type_list_;
};

class GE_FUNC_VISIBILITY OpTypeRegistrar {
 public:
  explicit OpTypeRegistrar(const std::string &op_type) noexcept { OpTypeContainer::Instance().Register(op_type); }
  ~OpTypeRegistrar() {}
};

#define REGISTER_OPTYPE_DECLARE(var_name, str_name) \
  FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const char_t *var_name

#define REGISTER_OPTYPE_DEFINE(var_name, str_name)           \
  const char_t *var_name = str_name;                         \
  const ge::OpTypeRegistrar g_##var_name##_reg(str_name)

#define IS_OPTYPE_EXISTING(str_name) (ge::OpTypeContainer::Instance().IsExisting(str_name))
}  // namespace ge

#endif  // INC_GRAPH_OP_TYPES_H_
