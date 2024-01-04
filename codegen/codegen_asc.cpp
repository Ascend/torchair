#ifndef CODEGEN_ASC_OPS_H
static_assert(0, "CODEGEN_ASC_OPS_H must be defined");
#else
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define OPS_DEF STR(CODEGEN_ASC_OPS_H)
#endif

#include <fstream>
#include OPS_DEF

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <output_dir>" << kEnd;
    return 1;
  }
  std::fstream file;
  file.open(argv[1], std::ios::out);

  Code code;
  code << "#ifndef __ASCIR_OPS_H__" << kEnd;
  code << "#define __ASCIR_OPS_H__" << kEnd;
  code << "#include \"ascir.h\"" << kEnd << kEnd << kEnd;

  for (auto &def : OpDef::defs) {
    bool supported = [&def]() {
      if (!def.Err().empty()) {
        return false;
      }
      for (auto &input : def.inputs) {
        if (input.is_dynamic || input.is_optional) {
          return false;
        }
      }
      return true;
    }();
    if (!supported) {
      continue;
    }

    std::stringstream op_reg;
    std::stringstream op_asc;
    op_reg << "REG_OPS(" << def.op << ")" << kEnd;
    op_asc << "namespace ascir::ops {" << kEnd;
    op_asc << "struct " << def.op << " : public ge::op::" << def.op << " {" << kEnd;
    op_asc << "union {" << kEnd;
    op_asc << "ge::Operator *__op;" << kEnd;
    for (size_t i = 0U; i < def.inputs.size(); i++) {
      std::string name = "x" + std::to_string(i + 1);
      if (def.inputs.size() == 1U) {
        name = "x";
      }
      op_reg << ".INPUT(" << name << ", TensorType::ALL())" << kEnd;
      op_asc << "ascir::OperatorInput<" << i << "> " << name << ";" << kEnd;
    }
    for (size_t i = 0U; i < def.outputs.size(); i++) {
      std::string name = "y" + std::to_string(i + 1);
      if (def.outputs.size() == 1U) {
        name = "y";
      }
      op_reg << ".OUTPUT(" << name << ", TensorType::ALL())" << kEnd;
      op_asc << "ascir::OperatorOutput<" << i << "> " << name << ";" << kEnd;
    }
    op_reg << "END_OPS(" << def.op << ");" << kEnd;
    op_asc << "};" << kEnd;

    op_asc << "static constexpr const char *Type = \"" << def.op << "\"" << kEnd;
    op_asc << "ascir::NodeAttr attr;" << kEnd;
    op_asc << "inline " << def.op << "(const char *name) : ge::op::" << def.op << "(name), __op(this), attr(*this) {}"
           << kEnd;
    op_asc << "};" << kEnd;
    op_asc << "}  // namespace ascir::ops" << kEnd;

    code << op_reg.str() << kEnd;
    code << op_asc.str() << kEnd;
  }
  code << "#endif // __ASCIR_OPS_H__" << kEnd;
  file << code.str() << kEnd;
  file.close();
}