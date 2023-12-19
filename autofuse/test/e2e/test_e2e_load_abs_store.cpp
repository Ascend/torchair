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

  std::vector<ascir::ImplGraph> test_impl_graphs = {ascir::ImplGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  auto kernel_code = codegen.GenerateKernel(test_impl_graphs);
  std::cout << kernel_code << std::endl;

  GTEST_SKIP() << "Compare kernel code here";
}
