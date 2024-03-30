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

#ifndef AUTOFUSE_ASCIR_REGISTRY_H
#define AUTOFUSE_ASCIR_REGISTRY_H
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include <sstream>
namespace ascir {
// todo c++的类ABI兼容性不好，后面考虑换成C接口实现
struct AscIrAttrDef {
  std::string name;
  std::string asc_ir_type;
  std::string ge_ir_type;
};
struct AscIrDef {
  using CodeGenerator = std::function<void(const AscIrDef &def, std::stringstream &ss)>;

  std::string file_path;
  int64_t line;
  std::string type;

  // 当前只有必选输入一种，没有其他类型，因此暂时简单处理，后续有复杂的optional后，defs的类型就不是string了
  std::vector<std::string> input_defs;
  std::vector<std::string> output_defs;
  std::vector<AscIrAttrDef> attr_defs;

  bool start_node;
  CodeGenerator infer_data_type_generator;
};
inline void SameDataTypeFromInput(const AscIrDef &def, std::stringstream &ss, const char *input_name) {
  for (const auto &output_def : def.output_defs) {
    ss << "  op." << output_def << ".dtype = static_cast<ge::DataType>(" << input_name << "_in.dtype);" << std::endl;
  }
}
inline void SameDataTypeFromFirstInput(const AscIrDef &def, std::stringstream &ss) {
  SameDataTypeFromInput(def, ss, def.input_defs[0].c_str());
}

class AscirRegistry {
 public:
  static AscirRegistry &GetInstance();
  void RegisterAscIr(const std::string &type, const AscIrDef &def);

  const std::unordered_map<std::string, AscIrDef> &GetAll() const;

 private:
  std::unordered_map<std::string, AscIrDef> types_to_ascir_;
};
}  // namespace ascir

#endif  // AUTOFUSE_ASCIR_REGISTRY_H
