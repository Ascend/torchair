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

#include "ascir_register.h"
#include "ascir_registry.h"
#include "graph/types.h"

namespace ascir {
AscirRegister::AscirRegister(const char *type, const char *def_file_path, int64_t line)
    : ir_def_{} {
  ir_def_.type = type;
  ir_def_.file_path = def_file_path;
  ir_def_.line = line;
  ir_def_.start_node = false;
}

AscirRegister &AscirRegister::Inputs(std::vector<std::string> &&input_names) {
  ir_def_.input_defs = std::move(input_names);
  return *this;
}
AscirRegister &AscirRegister::Outputs(std::vector<std::string> &&output_names) {
  ir_def_.output_defs = std::move(output_names);
  return *this;
}
AscirRegister::AscirRegister(const AscirRegister &other) {
  AscirRegistry::GetInstance().RegisterAscIr(other.ir_def_.type, other.ir_def_);
}
AscirRegister &AscirRegister::Attr(std::string name, std::string asc_type, std::string ge_type) {
  ir_def_.attr_defs.emplace_back(AscIrAttrDef{std::move(name), std::move(asc_type), std::move(ge_type)});
  return *this;
}
AscirRegister &AscirRegister::StartNode() {
  ir_def_.start_node = true;
  return *this;
}
AscirRegister &AscirRegister::InferDataType(AscIrDef::CodeGenerator infer_data_type_generator) {
  ir_def_.infer_data_type_generator = std::move(infer_data_type_generator);
  return *this;
}
template <>
AscirRegister &AscirRegister::Attr<ge::DataType>(std::string name) {
  return Attr(std::move(name), "ge::DataType", "Int");
}
}  // namespace ascir
