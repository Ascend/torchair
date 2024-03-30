/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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

#ifndef AUTOFUSE_ASCIR_REGISTER_H
#define AUTOFUSE_ASCIR_REGISTER_H
#include <string>
#include <vector>
#include "ascir_registry.h"
namespace ascir {
class AscirRegister {
 public:
  AscirRegister(const char *type, const char *def_file_path, int64_t line);
  AscirRegister &Inputs(std::vector<std::string> &&input_names);
  AscirRegister &Outputs(std::vector<std::string> &&output_names);

  template <class T>
  AscirRegister &Attr(std::string name);

  AscirRegister &InferDataType(AscIrDef::CodeGenerator infer_data_type_generator);
  AscirRegister &UseFirstInputDataType() {
    return InferDataType(SameDataTypeFromFirstInput);
  }

  AscirRegister &StartNode();

  AscirRegister(const AscirRegister &);
  AscirRegister &operator=(const AscirRegister &) = delete;

  AscirRegister(AscirRegister&&) noexcept = delete;
  AscirRegister &operator=(AscirRegister &&) noexcept = delete;

 private:
  AscirRegister &Attr(std::string name, std::string asc_type, std::string ge_type);

 private:
  AscIrDef ir_def_;
};

#define REG_ASC_IR(type) static auto g_register_##type = AscirRegister(#type, __FILE__, __LINE__)
#define REG_ASC_IR_START_NODE(type) REG_ASC_IR(type).Inputs({}).Outputs({"y"}).StartNode()
#define REG_ASC_IR_1IO(type) REG_ASC_IR(type).Inputs({"x"}).Outputs({"y"})
#define REG_ASC_IR_2I1O(type) REG_ASC_IR(type).Inputs({"x1", "x2"}).Outputs({"y"})

#define EXPORT_GENERATOR() void PreventOptimizeGenerator() { \
  extern void PreventLinkerOptimizeForAscIrGenerator(); \
  PreventLinkerOptimizeForAscIrGenerator(); \
  }
}

#endif  // AUTOFUSE_ASCIR_REGISTER_H
