#include "codegen_tiling_data.h"

#include <sstream>

codegen::TilingData::TilingData(const std::string& kernel_name, const std::string &class_name)
    : kernel_name(kernel_name), class_name(class_name) {}

const std::string codegen::TilingData::MacrosAndIncludes = {
  "#ifdef __CCE_KT_TEST__\n"
  "#include <stdint.h>\n"
  "#define BEGIN_TILING_DATA_DEF(name) struct name {\n"
  "#define TILING_DATA_FIELD_DEF(type, name) \\\n"
  "  type name; \\\n"
  "  inline void set_##name(type value) { name = value; } \\\n"
  "  inline type get_##name() { return name; }\n"
  "#define END_TILING_DATA_DEF };\n"
  "#define REGISTER_TILING_DATA_CLASS(op_type, tiling_type)\n"
  "#else\n"
  "#include \"register/tilingdata_base.h\"\n"
  "#endif\n"
};

std::string codegen::TilingData::Generate(const ascir::ImplGraph &graph) {
    std::stringstream ss;

    ss << MacrosAndIncludes;
    ss << std::endl;

    ss << "namespace optiling {" << std::endl;

    ss << this->ClassBegin() << std::endl;
    ss << "  TILING_DATA_FIELD_DEF(uint32_t, block_dim);" << std::endl;
    ss << "  TILING_DATA_FIELD_DEF(uint32_t, tiling_case);" << std::endl;
    for (auto size : graph.size_var()) {
        ss << "  " << this->DataFieldDefine(size) << std::endl;
    }
    ss << this->ClassEnd() << std::endl;
    ss << std::endl;

    ss << this->ClassRegister() << std::endl;
    ss << "}" << std::endl;

    return ss.str();
}

std::string codegen::TilingData::ClassBegin() {
  std::stringstream ss;
  ss << "BEGIN_TILING_DATA_DEF(" << this->class_name << ")";
  return ss.str();
}

std::string codegen::TilingData::DataFieldDefine(const ascir::SizeVar &size) {
  std::stringstream ss;
  ss << "TILING_DATA_FIELD_DEF(uint32_t, " << size.name << ");";
  return ss.str();
}

std::string codegen::TilingData::ClassEnd() {
  std::stringstream ss;
  ss << "END_TILING_DATA_DEF;";
  return ss.str();
}

std::string codegen::TilingData::ClassRegister() {
  std::stringstream ss;
  ss << "REGISTER_TILING_DATA_CLASS(" << this->kernel_name << ", " << this->class_name << ")";
  return ss.str();
}
