#include "e2e_load_abs_store.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "ascir_utils.h"
#include "optimize.h"
#include "codegen.h"

class E2E_LoadAbsStore : public ::testing::Test {
 protected:
  optimize::Optimizer optimizer;
  codegen::Codegen codegen;

  E2E_LoadAbsStore() : optimizer(optimize::OptimizerOptions{}), codegen(codegen::CodegenOptions{}) {}
};

TEST_F(E2E_LoadAbsStore, ConstructGraphWithAscir) {
  ascir::HintGraph test_graph("test_load_abs_store");
  LoadAbsStore_BeforeAutofuse(test_graph);
  GTEST_SKIP() << "Compare expect graph ir info here";
}

TEST_F(E2E_LoadAbsStore, InferOutput) {
  ascir::HintGraph test_graph("test_load_abs_store");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  ascir::HintGraph expected_graph("test_load_abs_store");
  LoadAbsStore_BeforeAutofuse(expected_graph);
  this->optimizer.InferOutput(expected_graph);

  EXPECT_EQ(ascir::utils::DebugHintGraphStr(test_graph), ascir::utils::DebugHintGraphStr(expected_graph));
}

TEST_F(E2E_LoadAbsStore, GetApiInfo) {
  ascir::HintGraph expect_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(expect_graph);

  ascir::ImplGraph expect_optimize_graph("expect_optimize_graph");
  expect_optimize_graph.CopyFrom(expect_graph);
  LoadAbsStore_AfterGetApiInfo(expect_optimize_graph);

  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);

  ascir::ImplGraph test_optimize_graph("test_optimize_graph");
  test_optimize_graph.CopyFrom(test_graph);
  optimizer.GetApiInfo(test_graph, test_optimize_graph);

  EXPECT_EQ(ascir::utils::DebugHintGraphStr(test_graph), ascir::utils::DebugHintGraphStr(expect_graph));
}

TEST_F(E2E_LoadAbsStore, AutoScheduler) {
  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  ascir::ImplGraph test_optimize_graph("test_optimize_graph");
  test_optimize_graph.CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_optimize_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs;
  optimizer.AutoScheduler(test_graph, test_optimize_graph, test_impl_graphs);
  auto& result_graph = test_impl_graphs[1];

  ascir::HintGraph expect_graph("expect_graph");
  LoadAbsStore_BeforeAutofuse(expect_graph);
  LoadAbsStore_AfterInferOutput(expect_graph);

  ascir::ImplGraph expect_optimize_graph("expect_optimize_graph");
  expect_optimize_graph.CopyFrom(expect_graph);
  LoadAbsStore_AfterGetApiInfo(expect_optimize_graph);

  ascir::ImplGraph expect_impl_graph(result_graph.GetName().c_str());
  expect_impl_graph.CopyFrom(expect_optimize_graph);
  LoadAbsStore_AfterScheduler(expect_impl_graph);

  EXPECT_EQ(ascir::utils::DebugHintGraphStr(expect_impl_graph), ascir::utils::DebugHintGraphStr(result_graph));
}

TEST_F(E2E_LoadAbsStore, BufQueAlloc) {
  ascir::HintGraph expect_graph("expect_graph");
  LoadAbsStore_BeforeAutofuse(expect_graph);
  LoadAbsStore_AfterInferOutput(expect_graph);

  ascir::ImplGraph expect_optimize_graph("expect_optimize_graph");
  expect_optimize_graph.CopyFrom(expect_graph);
  LoadAbsStore_AfterGetApiInfo(expect_optimize_graph);

  ascir::ImplGraph expect_impl_graph("test_impl_graph");
  expect_impl_graph.CopyFrom(expect_optimize_graph);
  LoadAbsStore_AfterScheduler(expect_impl_graph);
  LoadAbsStore_AfterQueBufAlloc(expect_impl_graph);

  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs = {ascir::ImplGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);

  optimizer.BufQueAlloc(test_graph, test_impl_graphs);

  EXPECT_EQ(ascir::utils::DebugHintGraphStr(expect_impl_graph), ascir::utils::DebugHintGraphStr(test_impl_graphs[0]));
}

TEST_F(E2E_LoadAbsStore, Codegen_Proto)
{
  ascir::HintGraph test_graph("LoadAbsStore");
  LoadAbsStore_BeforeAutofuse(test_graph);

  auto proto_code = codegen.GenerateProto(test_graph);
  std::cout << proto_code << std::endl;

  auto result = nlohmann::json::parse(proto_code);
  auto op = result;
  EXPECT_EQ(op["op"], "LoadAbsStore");
  EXPECT_EQ(op["language"], "cpp");

  auto input = op["input_desc"][0];
  EXPECT_EQ(input["name"], "x");
  EXPECT_EQ(input["param_type"], "required");
  EXPECT_EQ(input["type"][0], "fp16");
  EXPECT_EQ(input["format"][0], "ND");

  auto output = op["output_desc"][0];
  EXPECT_EQ(output["name"], "y");
  EXPECT_EQ(output["param_type"], "required");
  EXPECT_EQ(output["type"][0], "fp16");
  EXPECT_EQ(output["format"][0], "ND");
}

TEST_F(E2E_LoadAbsStore, Codegen_TilingData)
{
  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs = {ascir::ImplGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  auto tiling_data_code = codegen.GenerateTilingData(test_graph, test_impl_graphs);
  EXPECT_EQ(tiling_data_code, string{
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
    "\n"
    "namespace optiling {\n"
    "BEGIN_TILING_DATA_DEF(TilingData)\n"
    "  TILING_DATA_FIELD_DEF(uint32_t, block_dim);\n"
    "  TILING_DATA_FIELD_DEF(uint32_t, tiling_case);\n"
    "  TILING_DATA_FIELD_DEF(uint32_t, s0);\n"
    "  TILING_DATA_FIELD_DEF(uint32_t, s1);\n"
    "  TILING_DATA_FIELD_DEF(uint32_t, s2);\n"
    "  TILING_DATA_FIELD_DEF(uint32_t, z1t_size);\n"
    "  TILING_DATA_FIELD_DEF(uint32_t, z0z1Tb_size);\n"
    "END_TILING_DATA_DEF;\n"
    "\n"
    "REGISTER_TILING_DATA_CLASS(test_graph, TilingData)\n"
    "}\n"
    });
}

TEST_F(E2E_LoadAbsStore, Codegen_Tiling)
{
  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs = {ascir::ImplGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  auto tiling_code = codegen.GenerateTiling(test_graph, test_impl_graphs);
  EXPECT_EQ(tiling_code, std::string{
      "#include <stdexcept>\n"
      "#include \"test_graph_tiling.h\"\n"
      "#ifndef __CCE_KT_TEST__\n"
      "#include \"register/op_def_registry.h\"\n"
      "#endif\n"
      "\n"
      "extern \"C\" void GetTiling(optiling::TilingData& tiling_data) {\n"
      "  throw std::runtime_error(\"GetTiling Not Implemented\");\n"
      "}\n"
      "\n"
      "#ifndef __CCE_KT_TEST__\n"
      "namespace optiling {\n"
      "static ge::graphStatus TilingFunc(gert::TilingContext* context)\n"
      "{\n"
      "  TilingData tiling;\n"
      "  const gert::Shape& size_var_shape = context->GetInputShape(0)->GetOriginShape();\n"
      "  tiling.set_block_dim(48);\n"
      "  tiling.set_s0(size_var_shape.GetDim(0));\n"
      "  tiling.set_s1(size_var_shape.GetDim(1));\n"
      "  tiling.set_s2(size_var_shape.GetDim(2));\n"
      "\n"
      "  GetTiling(tiling);\n"
      "  context->SetBlockDim(tiling.get_block_dim());\n"
      "  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());\n"
      "  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());\n"
      "\n"
      "  return ge::GRAPH_SUCCESS;\n"
      "}\n"
      "}\n"
      "\n"
      "namespace ge {\n"
      "static ge::graphStatus InferShape(gert::InferShapeContext* context)\n"
      "{\n"
      "    return GRAPH_SUCCESS;\n"
      "}\n"
      "}\n"
      "\n"
      "namespace ops {\n"
      "class test_graph : public OpDef {\n"
      "public:\n"
      "    explicit test_graph(const char* name) : OpDef(name)\n"
      "    {\n"
      "        this->Input(\"x\")\n"
      "            .ParamType(REQUIRED)\n"
      "            .DataType({ge::DT_FLOAT16})\n"
      "            .Format({ge::FORMAT_ND})\n"
      "            .UnknownShapeFormat({ge::FORMAT_ND});\n"
      "        this->Output(\"y\")\n"
      "            .ParamType(REQUIRED)\n"
      "            .DataType({ge::DT_FLOAT16})\n"
      "            .Format({ge::FORMAT_ND})\n"
      "            .UnknownShapeFormat({ge::FORMAT_ND});\n"
      "\n"
      "        this->SetInferShape(ge::InferShape);\n"
      "        this->AICore().SetTiling(optiling::TilingFunc);\n"
      "        this->AICore().AddConfig(\"ascend910b\");\n"
      "    }\n"
      "};\n"
      "\n"
      "OP_ADD(test_graph);\n"
      "}\n"
      "\n"
      "#endif\n"
  });
}
