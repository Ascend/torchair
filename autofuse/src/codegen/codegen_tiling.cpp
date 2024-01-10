#include "codegen_tiling.h"

#include "dlfcn.h"
#include <sstream>
#include "codegen_common.h"

using namespace ascir;
using namespace codegen;

TilingLib::TilingLib(const std::string &lib_path, const std::string &codegen_symbol_name) {
  this->codegen_func_ = nullptr;

  if (lib_path.empty() || codegen_symbol_name.empty()) {
    return;
  }

  auto handle = dlopen(lib_path.c_str(), RTLD_LAZY);
  if (!handle) {
      std::cerr << "TilingLib open from " << lib_path << "failed. " << dlerror();
      return;
  }

  auto func = dlsym(handle, codegen_symbol_name.c_str());
  if (!func) {
    dlclose(handle);
    std::cerr << "Get codegen function from " << codegen_symbol_name << "failed. " << dlerror();
    return;
  }

  this->codegen_func_ = reinterpret_cast<TilingLibCodegenFunc>(func);
}

std::string TilingLib::Generate(const HintGraph &graph, const std::vector<ImplGraph> &impl_graphs) const {
  std::stringstream ss;

  ss << "#include \"" << CamelToLowerSneak(graph.GetName()) << "_tiling.h\"" << std::endl;
  ss << "#ifndef __CCE_KT_TEST__" << std::endl;
  ss << "#include \"register/op_def_registry.h\"" << std::endl;
  ss << "#endif" << std::endl;
  ss << std::endl;

  if (this->codegen_func_ != nullptr) {
    ss << this->codegen_func_(impl_graphs);
  } else {
    ss << "extern \"C\" void GetTiling(optiling::TilingData& tiling_data) {" << std::endl;
    ss << "  throw std::runtime_error(\"GetTiling Not Implemented\");" << std::endl;
    ss << "}" << std::endl;
  }
  ss << std::endl;

  ss << "#ifndef __CCE_KT_TEST__" << std::endl;

  ss << TilingFuncDef(graph);
  ss << std::endl;

  ss << InferShapeDef(graph);
  ss << std::endl;

  ss << OpDef(graph);
  ss << std::endl;

  ss << "#endif" << std::endl;

  return ss.str();
}

std::string TilingLib::TilingFuncDef(const HintGraph &graph) const {
  std::stringstream ss;

  ss << "namespace optiling {" << std::endl;

  ss << "static ge::graphStatus TilingFunc(gert::TilingContext* context)" << std::endl;
  ss << "{" << std::endl;
  ss << "  TilingData tiling;" << std::endl;

  // Use first input shape pass all size variable value
  ss << "  const gert::Shape& size_var_shape = context->GetInputShape(0)->GetOriginShape();" << std::endl;
  ss << "  tiling.set_block_dim(48);" << std::endl; // Only consider 48 for now

  int index = 0;
  for (auto size : graph.size_var()) {
    ss << "  tiling.set_" << size.name << "(size_var_shape.GetDim(" << index << "));" << std::endl;
    index++;
  }
  ss << std::endl;

  ss << "  GetTiling(tiling);" << std::endl;
  ss << "  context->SetBlockDim(tiling.get_block_dim());" << std::endl;
  ss << "  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());" << std::endl;;
  ss << "  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());" << std::endl;
  ss << std::endl;

  ss << "  return ge::GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;

  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::InferShapeDef(const HintGraph &graph) const {
  std::stringstream ss;

  ss << "namespace ge {" << std::endl;
  ss << "static ge::graphStatus InferShape(gert::InferShapeContext* context)" << std::endl;
  ss << "{" << std::endl;
  ss << "    return GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::OpDef(const HintGraph &graph) const {
  std::stringstream ss;

  ss << "namespace ops {" << std::endl;
  ss << "class " << graph.GetName() << " : public OpDef {" << std::endl;
  ss << "public:" << std::endl;
  ss << "    explicit " << graph.GetName() << "(const char* name) : OpDef(name)" << std::endl;
  ss << "    {" << std::endl;

  for (auto input : graph.GraphInputs()) {
    ss << OpInputDef(input);
  }
  for (auto output : graph.GraphOutputs()) {
    ss << OpOutputDef(output);
  }
  ss << std::endl;

  ss << "        this->SetInferShape(ge::InferShape);" << std::endl;
  ss << "        this->AICore().SetTiling(optiling::TilingFunc);" << std::endl;
  ss << "        this->AICore().AddConfig(\"ascend910b\");" << std::endl;
  ss << "    }" << std::endl;
  ss << "};" << std::endl;
  ss << std::endl;

  ss << "OP_ADD(" << graph.GetName() << ");" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

static const std::string GeDtypeName(ge::DataType dtype) {
  const std::map<ge::DataType, std::string> dtype_map = {
      {ge::DT_FLOAT, "ge::DT_FLOAT"},
      {ge::DT_FLOAT16, "ge::DT_FLOAT16"}
  };

  auto it = dtype_map.find(dtype);
  if (it == dtype_map.end()) {
    throw std::runtime_error("Unsupported dtype " + std::to_string(dtype));
  }

  return it->second;
}

std::string TilingLib::OpInputDef(const NodeView &node) const {
  std::stringstream ss;

  ss << "        this->Input(\"" << node->GetName() << "\")" << std::endl;
  ss << "            .ParamType(REQUIRED)" << std::endl;
  ss << "            .DataType({" << GeDtypeName(node.outputs()[0].dtype) << "})" << std::endl;
  ss << "            .Format({ge::FORMAT_ND})" << std::endl;
  ss << "            .UnknownShapeFormat({ge::FORMAT_ND});" << std::endl;

  return ss.str();
}

std::string TilingLib::OpOutputDef(const NodeView &node) const {
  std::stringstream ss;

  ss << "        this->Output(\"" << node->GetName() << "\")" << std::endl;
  ss << "            .ParamType(REQUIRED)" << std::endl;
  ss << "            .DataType({" << GeDtypeName(node.inputs[0]->dtype) << "})" << std::endl;
  ss << "            .Format({ge::FORMAT_ND})" << std::endl;
  ss << "            .UnknownShapeFormat({ge::FORMAT_ND});" << std::endl;

  return ss.str();
}
