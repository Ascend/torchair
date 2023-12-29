#include "e2e_load_abs_store.h"

#include "gtest/gtest.h"

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

TEST_F(E2E_LoadAbsStore, GetApiInfo) {
  ascir::HintGraph expect_graph("expect_graph");
  LoadAbsStore_BeforeAutofuse(expect_graph);

  ascir::ImplGraph expect_optimize_graph("expect_optimize_graph");
  expect_optimize_graph.CopyFrom(expect_graph);
  LoadAbsStore_AfterGetApiInfo(expect_optimize_graph);

  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);

  ascir::ImplGraph test_optimize_graph("test_optimize_graph");
  test_optimize_graph.CopyFrom(test_graph);
  optimizer.GetApiInfo(test_graph, test_optimize_graph);

  GTEST_SKIP() << "Compare Api info here";
}

TEST_F(E2E_LoadAbsStore, AutoScheduler) {
  ascir::HintGraph expect_graph("expect_graph");
  LoadAbsStore_BeforeAutofuse(expect_graph);

  ascir::ImplGraph expect_optimize_graph("expect_optimize_graph");
  expect_optimize_graph.CopyFrom(expect_graph);
  LoadAbsStore_AfterGetApiInfo(expect_optimize_graph);

  ascir::ImplGraph expect_impl_graph("expect_impl_graph");
  expect_impl_graph.CopyFrom(expect_optimize_graph);
  LoadAbsStore_AfterScheduler(expect_impl_graph);

  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);

  ascir::ImplGraph test_optimize_graph("test_optimize_graph");
  test_optimize_graph.CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_optimize_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs;
  optimizer.AutoScheduler(test_graph, test_optimize_graph, test_impl_graphs);

  GTEST_SKIP() << "Compare scheduler info here";
}

TEST_F(E2E_LoadAbsStore, BufQueAlloc) {
  ascir::HintGraph expect_graph("expect_graph");
  LoadAbsStore_BeforeAutofuse(expect_graph);

  ascir::ImplGraph expect_optimize_graph("expect_optimize_graph");
  expect_optimize_graph.CopyFrom(expect_graph);
  LoadAbsStore_AfterGetApiInfo(expect_optimize_graph);

  ascir::ImplGraph expect_impl_graph("expect_impl_graph");
  expect_impl_graph.CopyFrom(expect_optimize_graph);
  LoadAbsStore_AfterScheduler(expect_impl_graph);
  LoadAbsStore_AfterQueBufAlloc(expect_impl_graph);

  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs = {ascir::ImplGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);

  optimizer.BufQueAlloc(test_graph, test_impl_graphs);

  GTEST_SKIP() << "Compare buf/que info here";
}

TEST_F(E2E_LoadAbsStore, Codegen_Proto)
{
  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);

  auto proto_code = codegen.GenerateProto(test_graph);
  std::cout << proto_code << std::endl;

  GTEST_SKIP() << "Compare proto code here";
}

TEST_F(E2E_LoadAbsStore, Codegen_TilingData)
{
  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs = {ascir::ImplGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  auto tiling_data_code = codegen.GenerateTilingData(test_impl_graphs);
  std::cout << tiling_data_code << std::endl;

  GTEST_SKIP() << "Compare tiling data code here";
}

TEST_F(E2E_LoadAbsStore, Codegen_Tiling)
{
  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs = {ascir::ImplGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  auto tiling_code = codegen.GenerateTiling(test_impl_graphs);
  std::cout << tiling_code << std::endl;

  GTEST_SKIP() << "Compare tiling code here";
}

TEST_F(E2E_LoadAbsStore, Codegen_Kernel)
{
  ascir::HintGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs = {ascir::ImplGraph("load_abs_store")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  auto kernel_code = codegen.GenerateKernel(test_impl_graphs);
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
    "}\n"
  });

  std::cout << kernel_code;
}
