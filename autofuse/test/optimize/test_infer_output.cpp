#include "gtest/gtest.h"

#include "ascir_ops.h"
#include "optimize.h"

using namespace ascir;

class TestInferOutput : public ::testing::Test {
 protected:
  optimize::Optimizer optimizer;

  TestInferOutput() : optimizer(optimize::OptimizerOptions{}) {}
};

TEST_F(TestInferOutput, LoadAndStore_WillUseInputDtype) {
  ascir::Graph test_graph("test");

  ops::Data x("x");
  x.y.dtype = ge::DT_FLOAT16;

  ops::Load load("load");
  load.x = x;

  ops::Store store("store");
  store.x = load;

  test_graph.SetInputs({x});

  optimizer.InferOutput(test_graph);

  auto load_node = test_graph.Find("load");
  EXPECT_EQ(load_node.outputs[0].dtype, ge::DT_FLOAT16);

  auto store_node = test_graph.Find("store");
  EXPECT_EQ(store_node.outputs[0].dtype, ge::DT_FLOAT16);
}

TEST_F(TestInferOutput, ElemwiseUniary_WillUseInputDtypeAndStride) {
  ascir::Graph test_graph("test");

  ops::Data x("x");
  x.y.dtype = ge::DT_FLOAT16;

  ops::Load load("load");
  load.x = x;
  load.y.axis = {1};
  load.y.repeats = {SizeExpr(2)};
  load.y.strides = {SizeExpr(3)};

  ops::Abs abs("abs");
  abs.x = load;

  test_graph.SetInputs({x});

  optimizer.InferOutput(test_graph);

  auto abs_node = test_graph.Find("abs");
  EXPECT_EQ(abs_node.outputs[0].dtype, ge::DT_FLOAT16);
  EXPECT_EQ(abs_node.outputs[0].axis[0], 1);
  EXPECT_EQ(abs_node.outputs[0].repeats[0].nums[0], 2);
  EXPECT_EQ(abs_node.outputs[0].strides[0].nums[0], 3);
}
