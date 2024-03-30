#include "gtest/gtest.h"

#include "ascir_ops.h"

#include "graph/operator_reg.h"

#include "test_util.h"

namespace ge {
REG_OP(TestOp).INPUT(x, BasicType()).OP_END_FACTORY_REG(TestOp);
};

TEST(AscirOps_Register, RegOp_WillCreateAscirOpFactory) {
  ge::op::TestOp op("test_op");

  ascir::Graph graph("test");
  graph.SetInputs({op});
}

TEST(AscirCast, DataType_Ok) {
  ascir::Graph graph("test_graph");

  auto data = ascir::cg::ContiguousData("test_op1", graph, ge::DT_FLOAT, {});
  auto cast = ascir::cg::Cast("cast", data.y, ge::DT_FLOAT16);
  graph.SetInputs({data});


  auto cast_node = ge::GraphUtilsEx::GetComputeGraph(graph)->FindNode("cast");
  ASSERT_NE(cast_node, nullptr);
  ASSERT_NE(cast_node->GetInDataAnchor(0), nullptr);
  ASSERT_NE(cast_node->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
  EXPECT_EQ(cast_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetIdx(), 0);
  EXPECT_EQ(cast_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "test_op1");

  auto op_desc = cast_node->GetOpDesc();
  // attr set ok
  AttrEq(op_desc, ascir::ops::Cast::ATTR_dst_type, ge::DT_FLOAT16);
  // output data type infer ok
  EXPECT_EQ(op_desc->GetOutputDescPtr(0)->GetDataType(), ge::DT_FLOAT16);
}

TEST(AscirOps_StartNode, Ok) {
  ascir::Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data = ascir::cg::Data("test_op", graph, ge::DT_FLOAT16,
                              {a.id, b.id, c.id},
                              {A, B, C},
                              {B*C, C, ascir::SizeExpr::One()});

  auto data_ret = graph.Find("test_op");

  EXPECT_EQ(data_ret.outputs[0].dtype, ge::DT_FLOAT16);
  EXPECT_EQ(data_ret.outputs[0].axis[0], a.id);
  EXPECT_EQ(data_ret.outputs[0].axis[1], b.id);
  EXPECT_EQ(data_ret.outputs[0].axis[2], c.id);
  EXPECT_EQ(data_ret.outputs[0].repeats[0], A);
  EXPECT_EQ(data_ret.outputs[0].repeats[1], B);
  EXPECT_EQ(data_ret.outputs[0].repeats[2], C);
  EXPECT_EQ(data_ret.outputs[0].strides[0], B*C);
  EXPECT_EQ(data_ret.outputs[0].strides[1], C);
  EXPECT_EQ(data_ret.outputs[0].strides[2], ascir::SizeExpr{});
}

TEST(AscirOps_ContiguousStartNode, Ok) {
  ascir::Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data = ascir::cg::ContiguousData("test_op", graph, ge::DT_FLOAT16, {a, b, c});

  auto data_ret = graph.Find("test_op");

  EXPECT_EQ(data_ret.outputs[0].dtype, ge::DT_FLOAT16);
  EXPECT_EQ(data_ret.outputs[0].axis[0], a.id);
  EXPECT_EQ(data_ret.outputs[0].axis[1], b.id);
  EXPECT_EQ(data_ret.outputs[0].axis[2], c.id);
  EXPECT_EQ(data_ret.outputs[0].repeats[0], A);
  EXPECT_EQ(data_ret.outputs[0].repeats[1], B);
  EXPECT_EQ(data_ret.outputs[0].repeats[2], C);
  EXPECT_EQ(data_ret.outputs[0].strides[0], B*C);
  EXPECT_EQ(data_ret.outputs[0].strides[1], C);
  EXPECT_EQ(data_ret.outputs[0].strides[2], ascir::SizeExpr{});
}

TEST(AscirOps_FlashSoftmaxInferDataType, Ok) {
  ascir::Graph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto B = graph.CreateSizeVar("B");
  auto C = graph.CreateSizeVar("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);

  auto data0 = ascir::cg::ContiguousData("data0", graph, ge::DT_INT32, {a, b, c});
  auto data1 = ascir::cg::ContiguousData("data1", graph, ge::DT_FLOAT16, {a, b, c});

  ascir::cg::FlashSoftmax("fs", data0.y, data1.y);

  auto fs = graph.Find("fs");
  EXPECT_EQ(fs.outputs[0].dtype, ge::DT_INT32);
  EXPECT_EQ(fs.outputs[1].dtype, ge::DT_INT32);
  EXPECT_EQ(fs.outputs[2].dtype, ge::DT_INT32);
}
