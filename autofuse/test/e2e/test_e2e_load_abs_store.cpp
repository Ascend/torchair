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
    "  TILING_DATA_FIELD_DEF(uint32_t, z0b_size);\n"
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

TEST_F(E2E_LoadAbsStore, Codegen_Kernel)
{
  ascir::HintGraph test_graph("LoadAbsStore");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs = {ascir::ImplGraph("LoadAbsStore_1")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  std::cout << ascir::utils::DebugImplGraphStr(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  auto kernel_code = codegen.GenerateKernel(test_graph, test_impl_graphs);
  EXPECT_EQ(kernel_code, std::string{
    "#ifdef __CCE_KT_TEST__\n"
    "#include \"tikicpulib.h\"\n"
    "#include \"load_abs_store_tiling.h\"\n"
    "#define GET_TILING_DATA(tiling_data, tiling) \\\n"
    "    optiling::TilingData& tiling_data = *(optiling::TilingData*)(tiling);\n"
    "#endif\n"
    "\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "namespace utils {\n"
    "template <typename T>\n"
    "constexpr inline __aicore__ T Max(const T a) {\n"
    "  return a;\n"
    "}\n"
    "\n"
    "template <typename T, typename... Ts>\n"
    "constexpr inline __aicore__ T Max(const T a, const Ts... ts) {\n"
    "  return a > Max(ts...) ? a : Max(ts...);\n"
    "}\n"
    "\n"
    "template <typename T, typename... Ts>\n"
    "constexpr inline __aicore__ T Sum(const T a, const Ts... ts) {\n"
    "  return (a + ... + ts);\n"
    "}\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ __aicore__ void load_abs_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {\n"
    "GET_TILING_DATA(t, tiling);\n"
    "\n"
    "int block_dim = GetBlockIdx();\n"
    "const int z0B = block_dim % (t.s0 / t.z0b_size); block_dim /= t.s0 / t.z0b_size;\n"
    "\n"
    "GlobalTensor<half> x_y;\n"
    "x_y.SetGlobalBuffer((__gm__ half*)x);\n"
    "GlobalTensor<half> store_y;\n"
    "store_y.SetGlobalBuffer((__gm__ half*)y);\n"
    "\n"
    "TPipe tpipe;\n"
    "\n"
    "const uint32_t load_y_size = (t.z1t_size - 1) * t.s2 + (t.s2 - 1) + 1;\n"
    "const uint32_t load_y_que_depth = 2;\n"
    "const uint32_t load_y_que_buf_num = 2;\n"
    "const uint32_t abs_y_size = (t.z1t_size - 1) * t.s2 + (t.s2 - 1) + 1;\n"
    "const uint32_t abs_y_que_depth = 2;\n"
    "const uint32_t abs_y_que_buf_num = 2;\n"
    "\n"
    "\n"
    "const uint32_t q0_size = utils::Max(load_y_size * sizeof(half));\n"
    "const uint32_t q0_depth = utils::Max(load_y_que_depth);\n"
    "const uint32_t q0_buf_num = utils::Max(load_y_que_buf_num);\n"
    "TQue<TPosition::VECIN, q0_depth> q0;\n"
    "tpipe.InitBuffer(q0, q0_buf_num, q0_size);\n"
    "\n"
    "const uint32_t q1_size = utils::Max(abs_y_size * sizeof(half));\n"
    "const uint32_t q1_depth = utils::Max(abs_y_que_depth);\n"
    "const uint32_t q1_buf_num = utils::Max(abs_y_que_buf_num);\n"
    "TQue<TPosition::VECOUT, q1_depth> q1;\n"
    "tpipe.InitBuffer(q1, q1_buf_num, q1_size);\n"
    "\n"
    "\n"
    "for (int z0b = 0; z0b < t.z0b_size; z0b++) {\n"
    "for (int z1T = 0; z1T < t.s1 / t.z1t_size; z1T++) {\n"
    "{\n"
    "LocalTensor<uint8_t> q0_buf = q0.AllocTensor<uint8_t>();\n"
    "LocalTensor<half> load_y;\n"
    "load_y.SetAddrWithOffset(q0_buf, 0);\n"
    "DataCopy(load_y[0], x_y[z0B * (t.s1 * t.s2 * t.z0b_size) + z0b * (t.s1 * t.s2) + z1T * (t.s2 * t.z1t_size)], load_y_size);\n"
    "q0.EnQue(q0_buf);\n"
    "}\n"
    "{\n"
    "LocalTensor<uint8_t> q0_buf = q0.DeQue<uint8_t>();\n"
    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
    "LocalTensor<half> load_y;\n"
    "load_y.SetAddrWithOffset(q0_buf, 0);\n"
    "LocalTensor<half> abs_y;\n"
    "abs_y.SetAddrWithOffset(q1_buf, 0);\n"
    "Abs(abs_y[0], load_y[0], load_y_size);\n"
    "q1.EnQue(q1_buf);\n"
    "q0.FreeTensor(q0_buf);\n"
    "}\n"
    "{\n"
    "LocalTensor<uint8_t> q1_buf = q1.DeQue<uint8_t>();\n"
    "LocalTensor<half> abs_y;\n"
    "abs_y.SetAddrWithOffset(q1_buf, 0);\n"
    "DataCopy(store_y[z0B * (t.s1 * t.s2 * t.z0b_size) + z0b * (t.s1 * t.s2) + z1T * (t.s2 * t.z1t_size)], abs_y[0], abs_y_size);\n"
    "q1.FreeTensor(q1_buf);\n"
    "}\n"
    "}\n"
    "}\n"
    "}\n"
  });
}
