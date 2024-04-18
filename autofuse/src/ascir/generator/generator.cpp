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
#include "generator.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "ascir_registry.h"
namespace ascir {
namespace {
const char *GetPureFileName(const char *path) {
  auto name = std::strrchr(path, '/');
  if (name == nullptr) {
    name = path;
  } else {
    ++name;
  }
  return name;
}

void GenAscIr(const AscIrDef &def, std::stringstream &ss) {
  if (def.attr_defs.empty()) {
    ss << "REG_OPS(" << def.type << ")" << std::endl;
  } else {
    ss << "REG_OPS_WITH_ATTR(" << def.type << ")" << std::endl;
  }

  // generate attr definitions
  const auto &attr_defs = def.attr_defs;
  if (!attr_defs.empty()) {
    ss << "  OPS_ATTR_NAME_START()" << std::endl;
    for (const auto &attr_def : attr_defs) {
      ss << "    OPS_ATTR_NAME(" << attr_def.name << ")" << std::endl;
    }
    ss << "  OPS_ATTR_NAME_END()" << std::endl;

    for (const auto &attr_def : attr_defs) {
      ss << "  OPS_ATTR(" << attr_def.name << ", " << attr_def.asc_ir_type << ")" << std::endl;
    }
  }

  // generate input output definitions
  const auto &input_defs = def.input_defs;
  for (size_t i = 0UL; i < input_defs.size(); ++i) {
    ss << "  OPS_INPUT(" << i << ", " << input_defs[i] << ")" << std::endl;
  }

  const auto &output_defs = def.output_defs;
  for (size_t i = 0UL; i < output_defs.size(); ++i) {
    ss << "  OPS_OUTPUT(" << i << ", " << output_defs[i] << ")" << std::endl;
  }

  ss << "END_OPS(" << def.type << ")" << std::endl;
}

void GenGeIr(const AscIrDef &def, std::stringstream &ss) {
  ss << "REG_OP(" << def.type << ")" << std::endl;

  const auto &attr_defs = def.attr_defs;
  if (!attr_defs.empty()) {
    for (const auto &attr_def : attr_defs) {
      ss << "    .REQUIRED_ATTR(" << attr_def.name << ", " << attr_def.ge_ir_type << ")" << std::endl;
    }
  }

  const auto &input_defs = def.input_defs;
  for (const auto &input_def : input_defs) {
    ss << "    .INPUT(" << input_def << ", TensorType::ALL())" << std::endl;
  }

  const auto &output_defs = def.output_defs;
  for (const auto &output_def : output_defs) {
    ss << "    .OUTPUT(" << output_def << ", TensorType::ALL())" << std::endl;
  }

  ss << ".OP_END_FACTORY_REG(" << def.type << ")" << std::endl;
}

class FunctionGenerator {
 public:
  explicit FunctionGenerator(const AscIrDef &def) : def_(def) {}
  virtual ~FunctionGenerator() = default;

  virtual void Gen(std::stringstream &ss) const {
    GenDefinition(ss);

    GenInstantiation(ss);
    ss << std::endl;

    if (GenConnectInputs(ss)) {
      ss << std::endl;
    }

    if (GenAttrAssignment(ss)) {
      ss << std::endl;
    }

    GenSchedInfo(ss);
    ss << std::endl;

    if (GenOutputsAssignment(ss)) {
      ss << std::endl;
    }

    GenPaddingAxis(ss);
    ss << std::endl;

    GenReturn(ss);
  }

  virtual void GenDefinition(std::stringstream &ss) const;
  virtual void GenInstantiation(std::stringstream &ss) const;
  virtual bool GenConnectInputs(std::stringstream &ss) const;
  virtual bool GenAttrAssignment(std::stringstream &ss) const;
  virtual void GenSchedInfo(std::stringstream &ss) const {
    ss << "  op.attr.sched.exec_order = GenNextExecId(op);" << std::endl;
    ss << "  SET_SCHED_AXIS_IF_IN_CONTEXT();" << std::endl;
  }
  virtual bool GenOutputsAssignment(std::stringstream &ss) const {
    bool generated = false;

    // generate infer data type code
    if (def_.infer_data_type_generator != nullptr) {
      generated = true;
      def_.infer_data_type_generator(def_, ss);
    }
    if (def_.infer_view_generator != nullptr) {
      generated = true;
      def_.infer_view_generator(def_, ss);
    }
    return generated;
  }
  virtual void GenPaddingAxis(std::stringstream &ss) const {
    for (const auto &name : def_.output_defs) {
      ss << "  THROW(PadOutputViewToSched(op." << name << "));" << std::endl;
    }
  }
  virtual void GenReturn(std::stringstream &ss) const {
    ss << "  return op;" << std::endl;
    ss << "}" << std::endl;
  }

 protected:
  const AscIrDef &def_;
};
void ascir::FunctionGenerator::GenDefinition(std::stringstream &ss) const {
  const std::vector<std::string> *input_defs;
  std::vector<std::string> empty_input_defs;

  if (def_.start_node) {
    // todo 由于历史原因，start_node（例如Data）仍然带有输入定义，但是这种输入实际是不连边的。
    //      但是为了最小化修改，当前先不修改Data的定义，后续需要做调整，对与StartNode类型，不定义输入，
    //      或者认为没有输入的op就是start node，在定义IR时不需要再显式指定start node标记
    input_defs = &empty_input_defs;
  } else {
    input_defs = &def_.input_defs;
  }

  // template <const int I1, const int I2>
  if (!input_defs->empty()) {
    ss << "template <";
    for (size_t i = 0UL; i < input_defs->size(); ++i) {
      if (i > 0UL) {
        ss << ", ";
      }
      ss << "const int I" << i;
    }
    ss << ">" << std::endl;
  } else {
    ss << "inline ";
  }

  // ascir::ops::OpType OpType(const char *name, const OperatorOutput<IndexN> &x_in, ...) {
  ss << "ascir::ops::" << def_.type << ' ' << def_.type << "(const char *name";
  if (!input_defs->empty()) {
    for (size_t i = 0UL; i < input_defs->size(); ++i) {
      const auto &input_def = input_defs->at(i);
      ss << ", const OperatorOutput<I" << i << "> &" << input_def << "_in";
    }
  } else {
    ss << ", ascir::Graph &graph";
  }

  for (const auto &attr_def : def_.attr_defs) {
    ss << ", const " << attr_def.asc_ir_type << " &" << attr_def.name;
  }
  ss << ") {" << std::endl;
}
void ascir::FunctionGenerator::GenInstantiation(std::stringstream &ss) const {
  ss << "  ascir::ops::" << def_.type << " op(name);" << std::endl;
  if (def_.start_node) {
    ss << "  graph.AddNode(op);" << std::endl;
  } else {
    ss << "  AddToGraphFollowOp(*(" << def_.input_defs[0] << "_in.__op), op);" << std::endl;
  }
}
bool ascir::FunctionGenerator::GenConnectInputs(std::stringstream &ss) const {
  // todo 这里与GenFunctionDefinition同理，后续删除
  if (def_.start_node) {
    return false;
  }
  if (!def_.input_defs.empty()) {
    for (const auto &input_def : def_.input_defs) {
      ss << "  op." << input_def << " = " << input_def << "_in;" << std::endl;
    }
  }
  return !def_.input_defs.empty();
}
bool ascir::FunctionGenerator::GenAttrAssignment(std::stringstream &ss) const {
  if (!def_.attr_defs.empty()) {
    for (const auto &attr_def : def_.attr_defs) {
      ss << "  op." << attr_def.name << " = " << attr_def.name << ';' << std::endl;
    }
  }
  return !def_.attr_defs.empty();
}

class StartNodeFuncGenerator : public FunctionGenerator {
 public:
  explicit StartNodeFuncGenerator(const AscIrDef &def) : FunctionGenerator(def) {}
  void Gen(std::stringstream &ss) const override {
    if (!def_.start_node || def_.output_defs.size() != 1UL) {
      return;
    }
    FunctionGenerator::Gen(ss);
  }
  void GenDefinition(std::stringstream &ss) const override {
    // inline ascir::ops::OpType OpType
    ss << "inline " << "ascir::ops::" << def_.type << ' ' << def_.type
       << "(const char *name, ascir::Graph &graph, ge::DataType dt"
       << ", const std::vector<ascir::AxisId> &axes"
       << ", const std::vector<ascir::SizeExpr> &repeats"
       << ", const std::vector<ascir::SizeExpr> &strides";

    for (const auto &attr_def : def_.attr_defs) {
      ss << ", const " << attr_def.asc_ir_type << " &" << attr_def.name;
    }
    ss << ") {" << std::endl;
  }
  bool GenOutputsAssignment(std::stringstream &ss) const override {
    const auto &output_name = def_.output_defs[0];
    ss << "  op." << output_name << ".dtype = dt;" << std::endl;
    ss << "  op." << output_name << ".axis = axes;" << std::endl;
    ss << "  op." << output_name << ".repeats = repeats;" << std::endl;
    ss << "  op." << output_name << ".strides = strides;" << std::endl;
    return true;
  }
};

class ContiguousStartNodeFuncGenerator : FunctionGenerator {
 public:
  explicit ContiguousStartNodeFuncGenerator(const AscIrDef &def) : FunctionGenerator(def) {}
  void Gen(std::stringstream &ss) const override {
    if (!def_.start_node || def_.output_defs.size() != 1UL) {
      return;
    }
    FunctionGenerator::Gen(ss);
  }
  void GenDefinition(std::stringstream &ss) const override {
    ss << "inline " << "ascir::ops::" << def_.type << " Contiguous" << def_.type
       << "(const char *name, ascir::Graph &graph, ge::DataType dt"
       << ", const std::vector<ascir::Axis> &axes";

    for (const auto &attr_def : def_.attr_defs) {
      ss << ", const " << attr_def.asc_ir_type << " &" << attr_def.name;
    }
    ss << ") {" << std::endl;
  }
  bool GenOutputsAssignment(std::stringstream &ss) const override {
    const auto &output_name = def_.output_defs[0];
    ss << "  op." << output_name << ".dtype = dt;" << std::endl;
    ss << "  op." << output_name << ".SetContiguousView(axes);" << std::endl;
    return true;
  }
};

void GetHeaderGuarderFromPath(const char *path, std::stringstream &ss) {
  auto name = GetPureFileName(path);

  ss << "ASCIR_OPS_";

  while (*name != '\0') {
    auto c = toupper(*name++);
    if (c < 'A' || c > 'Z') {
      ss << '_';
    } else {
      ss << (char)c;
    }
  }

  ss << '_';
}
}  // namespace

void GenAll(std::stringstream &ss) {
  std::stringstream ss_asc_ir;
  std::stringstream ss_ge_ir;

  ss << R"(#include "ascir_ops_utils.h")" << std::endl << std::endl;
  ss << R"(#include "cg_utils.h")" << std::endl << std::endl;

  std::map<std::pair<std::string, int64_t>, AscIrDef> ordered_keys_to_def;
  for (const auto &type_and_def : AscirRegistry::GetInstance().GetAll()) {
    ordered_keys_to_def[std::make_pair(type_and_def.second.file_path, type_and_def.second.line)] = type_and_def.second;
  }

  for (const auto &key_and_def : ordered_keys_to_def) {
    ss << "// Defined at " << GetPureFileName(key_and_def.second.file_path.c_str()) << ':' << key_and_def.second.line
       << std::endl;

    ss << "namespace ge {" << std::endl;
    GenGeIr(key_and_def.second, ss);
    ss << "}" << std::endl;  // namespace ge

    ss << "namespace ascir {" << std::endl;
    ss << "namespace ops {" << std::endl;
    GenAscIr(key_and_def.second, ss);
    ss << "}" << std::endl;               // namespace ops
    ss << "}" << std::endl << std::endl;  // namespace ascir
  }

  ss << "namespace ascir {" << std::endl;
  ss << "namespace cg {" << std::endl;
  for (const auto &key_and_def : ordered_keys_to_def) {
    FunctionGenerator(key_and_def.second).Gen(ss);
    StartNodeFuncGenerator(key_and_def.second).Gen(ss);
    ContiguousStartNodeFuncGenerator(key_and_def.second).Gen(ss);
  }
  ss << "}" << std::endl;               // namespace cg
  ss << "}" << std::endl << std::endl;  // namespace ascir
}

void GenHeaderFileToStream(const char *path, std::stringstream &ss) {
  std::stringstream ss_header_guarder;
  GetHeaderGuarderFromPath(path, ss_header_guarder);
  auto guarder = ss_header_guarder.str();

  ss << "// Generated from asc-ir definition files, "
        "any modification made to this file may be overwritten after compile."
     << std::endl;
  ss << "// If you want to add self-defined asc-ir, please create a seperated header file." << std::endl;
  ss << "#ifndef " << guarder << std::endl;
  ss << "#define " << guarder << std::endl << std::endl;

  GenAll(ss);

  ss << "#endif  // " << guarder << std::endl;
}

int GenHeaderFile(const char *path) {
  std::stringstream ss;
  GenHeaderFileToStream(path, ss);
  std::ofstream fs(path);
  if (!fs) {
    return -1;
  }
  fs << ss.str();
  fs.close();
  return 0;
}

void PreventLinkerOptimizeForAscIrGenerator() {
  GenHeaderFile("::memory::");
}
}  // namespace ascir