#include "gtest/gtest.h"

#include "graph/operator_reg.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_utils.h"

using namespace ascir;

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const int64_t &expect) {
  int64_t value = -1;
  ge::AttrUtils::GetInt(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const vector<int64_t> &expect) {
  vector<int64_t> value;
  ge::AttrUtils::GetListInt(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

template <typename T>
void AttrEq(T &holder, const std::string attr_name, const vector<vector<int64_t>> &expect) {
  vector<vector<int64_t>> value;
  ge::AttrUtils::GetListListInt(holder, attr_name, value);
  EXPECT_EQ(value, expect);
}

namespace ge {
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Data)
}

REG_OPS(Data)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Data)

TEST(TestAscir, AscirOperator_ShouldHas_Fields) {
  Data data("test_op");

  data.attr.sched.exec_order = 10;
  data.attr.sched.axis = {0, 1, 2};

  data.y.axis = {0, 1, 2};
  data.y.repeats = {SizeExpr({1}, {2}), SizeExpr({3}, {4})};
  data.y.strides = {SizeExpr({5, 6}, {7, 8}), SizeExpr({9, 10}, {11, 12})};

  ascir::Graph graph("test_graph");
  graph.SetInputs({data});
  auto result_op = ge::GraphUtilsEx::GetComputeGraph(graph)->FindNode("test_op")->GetOpDesc();

  AttrEq(result_op, NodeAttr::SCHED_EXEC_ORDER, 10);
  AttrEq(result_op, NodeAttr::SCHED_AXIS, {0, 1, 2});

  auto result_y = result_op->GetOutputDesc(0);
  AttrEq(result_y, data.y.AXIS, {0, 1, 2});
  AttrEq(result_y, data.y.repeats.NUM_OF_FACTOR, 2);
  AttrEq(result_y, data.y.repeats.NUMS, vector<vector<int64_t>>{{1}, {3}});
  AttrEq(result_y, data.y.repeats.DENS, vector<vector<int64_t>>{{2}, {4}});

  AttrEq(result_y, data.y.strides.NUM_OF_FACTOR, 2);
  AttrEq(result_y, data.y.strides.NUMS, vector<vector<int64_t>>{{5, 6}, {9, 10}});
  AttrEq(result_y, data.y.strides.DENS, vector<vector<int64_t>>{{7, 8}, {11, 12}});

  auto result_node = graph.Find("test_op");
  EXPECT_EQ(result_node.attr.sched.exec_order, 10);

  int axis_id = 0;
  for (auto axis : result_node.attr.sched.axis()) {
    EXPECT_EQ(axis, axis_id);
    axis_id++;
  }
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(result_node.attr.sched.axis[i], i);
  }

  EXPECT_EQ(result_node.outputs[0].axis[0], 0);
  EXPECT_EQ(result_node.outputs[0].axis[1], 1);
  EXPECT_EQ(result_node.outputs[0].axis[2], 2);

  auto stride = result_node.outputs[0].strides();
  EXPECT_EQ(stride[0].nums[0], 5);
  EXPECT_EQ(stride[0].nums[1], 6);
  EXPECT_EQ(stride[0].dens[0], 7);
  EXPECT_EQ(stride[0].dens[1], 8);
  EXPECT_EQ(stride[1].nums[0], 9);
  EXPECT_EQ(stride[1].nums[1], 10);
  EXPECT_EQ(stride[1].dens[0], 11);
  EXPECT_EQ(stride[1].dens[1], 12);

  EXPECT_EQ(result_node.outputs[0].repeats[1].nums[0], 3);
}

/** Create a graph ready for hold attributes */
inline ascir::Graph CreateTestGraph() {
  ascir::Graph graph("test_graph");
  ge::op::Data x("x");
  // Need using SetInputs to trigger initialization of graph
  // so that it can hold attributes
  graph.SetInputs({x});
  return std::move(graph);
}

TEST(Ascir_AxisOperations, SizeExpr_SupportZero) {
  SizeExpr zero = SizeExpr::Zero();
  EXPECT_TRUE(zero.is_zero);
}

TEST(Ascir_AxisOperations, SizeExpr_SupportCompareToOne) {
  SizeExpr one = SizeExpr::One();
  EXPECT_TRUE(one == 1);
}

TEST(Ascir_AxisOperations, SizeExpr_MulZero_WillBeZero) {
  SizeExpr result = SizeExpr::One() * SizeExpr::Zero();
  EXPECT_TRUE(result.is_zero);

  SizeExpr expr = SizeExpr::One();
  expr *= SizeExpr::Zero();
  EXPECT_TRUE(expr.is_zero);
}

TEST(Ascir_AxisOperations, SizeExpr_ZeroDivAny_WillBeZero) {
  SizeExpr result = SizeExpr::Zero() / SizeExpr::One();
  EXPECT_TRUE(result.is_zero);

  SizeExpr expr = SizeExpr::Zero();
  expr /= SizeExpr::One();
  EXPECT_TRUE(expr.is_zero);
}

TEST(Ascir_AxisOperations, SizeExpr_SupportConstructBy_SizeVar) {
  ascir::Graph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  SizeExpr expr({s0}, {s1});

  ASSERT_EQ(expr.nums.size(), 1);
  EXPECT_EQ(expr.nums[0], s0.id);
  ASSERT_EQ(expr.dens.size(), 1);
  EXPECT_EQ(expr.dens[0], s1.id);
}

TEST(Ascir_AxisOperations, SizeExpr_SupportConstructBy_Divider) {
  ascir::Graph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  SizeExpr expr = s0 / s1;

  ASSERT_EQ(expr.nums.size(), 1);
  EXPECT_EQ(expr.nums[0], s0.id);
  ASSERT_EQ(expr.dens.size(), 1);
  EXPECT_EQ(expr.dens[0], s1.id);

  SizeExpr expr2 = s0 / SizeExpr({s1});
  EXPECT_EQ(expr, expr2);
}

TEST(Ascir_AxisOperations, SizeExpr_SupportConstructBy_Multiplier) {
  ascir::Graph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  SizeExpr expr = s0 * s1;

  ASSERT_EQ(expr.nums.size(), 2);
  EXPECT_EQ(expr.nums[0], s0.id);
  EXPECT_EQ(expr.nums[1], s1.id);

  SizeExpr expr2 = s0 * SizeExpr({s1});
  EXPECT_EQ(expr, expr2);
}

TEST(Ascir, CanGetPeer_FromInput) {
  Data x1("x1"), x2("x2");

  x2.x = x1;
  x2.y.axis = {0, 1, 2};

  ascir::Graph graph("test_graph");
  graph.SetInputs({x1});

  auto result_x1 = graph.Find("x1");
  auto result_x2 = graph.Find("x2");
  EXPECT_EQ(result_x2.inputs[0]->Owner(), result_x1);
}

TEST(Ascir_AxisOperations, CreateSize_WillSetSizeTable_ToGraphAttr) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1", 100);

  EXPECT_EQ(s0.id, 0);
  EXPECT_EQ(s0.name, "s0");
  EXPECT_EQ(s0.type, ascir::SizeVar::SIZE_TYPE_VAR);
  EXPECT_EQ(s1.id, 1);
  EXPECT_EQ(s1.name, "s1");
  EXPECT_EQ(s1.type, ascir::SizeVar::SIZE_TYPE_CONST);
  EXPECT_EQ(s1.value, 100);

  auto result_graph = ge::GraphUtilsEx::GetComputeGraph(graph);

  vector<string> result_size_names;
  vector<int64_t> result_size_types;
  vector<int64_t> result_size_values;
  ge::AttrUtils::GetListStr(result_graph, graph.size_var.NAME, result_size_names);
  ge::AttrUtils::GetListInt(result_graph, graph.size_var.TYPE, result_size_types);
  ge::AttrUtils::GetListInt(result_graph, graph.size_var.VALUE, result_size_values);

  EXPECT_EQ(result_size_names[s0.id], "s0");
  EXPECT_EQ(result_size_types[s0.id], ascir::SizeVar::SIZE_TYPE_VAR);

  EXPECT_EQ(result_size_names[s1.id], "s1");
  EXPECT_EQ(result_size_types[s1.id], ascir::SizeVar::SIZE_TYPE_CONST);
  EXPECT_EQ(result_size_values[s1.id], 100);
}

TEST(Ascir_AxisOperations, CreateAxis_WillSetAxisTable_ToGraphAttr) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1", 100);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0 / s1);

  EXPECT_EQ(z0.id, 0);
  EXPECT_EQ(z0.name, "z0");
  EXPECT_EQ(z0.type, ascir::Axis::AXIS_TYPE_ORIGINAL);
  EXPECT_EQ(z0.size.nums, vector{s0.id});
  EXPECT_EQ(z0.size.dens, vector<ascir::SizeVarId>{});
  EXPECT_EQ(z0.from, vector<ascir::AxisId>{});

  EXPECT_EQ(z1.id, 1);
  EXPECT_EQ(z1.name, "z1");
  EXPECT_EQ(z1.type, ascir::Axis::AXIS_TYPE_ORIGINAL);
  EXPECT_EQ(z1.size.nums, vector{s0.id});
  EXPECT_EQ(z1.size.dens, vector{s1.id});
  EXPECT_EQ(z1.from, vector<ascir::AxisId>{});

  auto result_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  vector<string> result_axis_names;
  vector<ascir::Axis::Type> result_axis_types;
  vector<vector<ascir::SizeVarId>> result_axis_size_num;
  vector<vector<ascir::SizeVarId>> result_axis_size_den;
  vector<vector<ascir::AxisId>> result_axis_from;

  ge::AttrUtils::GetListStr(result_graph, graph.axis.NAME, result_axis_names);
  ge::AttrUtils::GetListInt(result_graph, graph.axis.TYPE, result_axis_types);
  ge::AttrUtils::GetListListInt(result_graph, graph.axis.SIZE_NUMS, result_axis_size_num);
  ge::AttrUtils::GetListListInt(result_graph, graph.axis.SIZE_DENS, result_axis_size_den);
  ge::AttrUtils::GetListListInt(result_graph, graph.axis.FROM, result_axis_from);

  EXPECT_EQ(z0.name, result_axis_names[z0.id]);
  EXPECT_EQ(z0.type, result_axis_types[z0.id]);
  EXPECT_EQ(z0.size.nums, result_axis_size_num[z0.id]);
  EXPECT_EQ(z0.size.dens, result_axis_size_den[z0.id]);
  EXPECT_EQ(z0.from, result_axis_from[z0.id]);

  EXPECT_EQ(z1.name, result_axis_names[z1.id]);
  EXPECT_EQ(z1.type, result_axis_types[z1.id]);
  EXPECT_EQ(z1.size.nums, result_axis_size_num[z1.id]);
  EXPECT_EQ(z1.size.dens, result_axis_size_den[z1.id]);
  EXPECT_EQ(z1.from, result_axis_from[z1.id]);
}

TEST(Ascir_AxisOperations, BlockSplit_WillCreate_BlockOutAndInAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  auto [z0_out, z0_in] = graph.BlockSplit(z0.id);

  auto result_z0_out = graph.axis[z0_out.id];
  auto result_z0_in = graph.axis[z0_in.id];
  EXPECT_EQ(result_z0_in.name, "z0b");
  EXPECT_EQ(result_z0_in.type, ascir::Axis::AXIS_TYPE_BLOCK_INNER);
  ASSERT_EQ(result_z0_in.from.size(), 1);
  EXPECT_EQ(result_z0_in.from[0], z0.id);
  ASSERT_EQ(result_z0_in.size.nums.size(), 1);
  EXPECT_EQ(result_z0_in.size.nums[0], 1);
  ASSERT_EQ(result_z0_in.size.dens.size(), 0);

  EXPECT_EQ(result_z0_out.name, "z0B");
  EXPECT_EQ(result_z0_out.type, ascir::Axis::AXIS_TYPE_BLOCK_OUTER);
  ASSERT_EQ(result_z0_out.from.size(), 1);
  EXPECT_EQ(result_z0_out.from[0], z0.id);
  EXPECT_EQ(result_z0_out.size.nums.size(), 1);
  EXPECT_EQ(result_z0_out.size.nums[0], s0.id);
  EXPECT_EQ(result_z0_out.size.dens.size(), 1);
  EXPECT_EQ(result_z0_out.size.dens[0], z0_in.size.nums[0]);

  auto block_size = result_z0_in.size.nums[0];
  EXPECT_EQ(graph.size_var[block_size].name, "z0b_size");
}

TEST(Ascir_AxisOperations, TileSplit_WillCreate_TileOutAndInAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", ascir::SizeExpr{{s0.id}});

  auto [z0_out, z0_in] = graph.TileSplit(z0.id);

  auto result_z0_out = graph.axis[z0_out.id];
  auto result_z0_in = graph.axis[z0_in.id];
  EXPECT_EQ(result_z0_in.name, "z0t");
  EXPECT_EQ(result_z0_in.type, ascir::Axis::AXIS_TYPE_TILE_INNER);
  ASSERT_EQ(result_z0_in.from.size(), 1);
  EXPECT_EQ(result_z0_in.from[0], z0.id);
  ASSERT_EQ(result_z0_in.size.nums.size(), 1);
  EXPECT_EQ(result_z0_in.size.nums[0], 1);
  ASSERT_EQ(result_z0_in.size.dens.size(), 0);

  EXPECT_EQ(result_z0_out.name, "z0T");
  EXPECT_EQ(result_z0_out.type, ascir::Axis::AXIS_TYPE_TILE_OUTER);
  ASSERT_EQ(result_z0_out.from.size(), 1);
  EXPECT_EQ(result_z0_out.from[0], z0.id);
  EXPECT_EQ(result_z0_out.size.nums.size(), 1);
  EXPECT_EQ(result_z0_out.size.nums[0], s0.id);
  EXPECT_EQ(result_z0_out.size.dens.size(), 1);
  EXPECT_EQ(result_z0_out.size.dens[0], z0_in.size.nums[0]);

  auto tile_size = result_z0_in.size.nums[0];
  EXPECT_EQ(graph.size_var[tile_size].name, "z0t_size");
}

TEST(Ascir_AxisOperations, MergeAxis_WillCreate_MergedAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", ascir::SizeExpr{{{s0.id}}, {{s1.id}}});
  auto z1 = graph.CreateAxis("z1", ascir::SizeExpr{{s2.id}, {s3.id}});

  auto z3 = graph.MergeAxis({z0.id, z1.id});

  auto result_z3 = graph.axis[z3.id];
  EXPECT_EQ(result_z3.name, "z0z1");
  EXPECT_EQ(result_z3.type, ascir::Axis::AXIS_TYPE_MERGED);
  EXPECT_EQ(result_z3.from.size(), 2);
  EXPECT_EQ(result_z3.from[0], z0.id);
  EXPECT_EQ(result_z3.from[1], z1.id);
  EXPECT_EQ(result_z3.size.nums.size(), 2);
  EXPECT_EQ(result_z3.size.nums[0], s0.id);
  EXPECT_EQ(result_z3.size.nums[1], s2.id);
  EXPECT_EQ(result_z3.size.dens.size(), 2);
  EXPECT_EQ(result_z3.size.dens[0], s1.id);
  EXPECT_EQ(result_z3.size.dens[1], s3.id);
}

TEST(Ascir_AxisOperations, ApplySplit_OnNode_WillSplitNodeAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", ascir::SizeExpr{{s0.id}});

  Data data("test_op");

  data.attr.sched.axis = {z0.id};
  data.y.axis = {z0.id};
  data.y.repeats = {SizeExpr{{s0.id}}};
  data.y.strides = {SizeExpr{}};
  graph.SetInputs({data});

  auto result_op = graph.Find("test_op");

  auto [z0_out, z0_in] = graph.TileSplit(z0.id);
  graph.ApplySplit(result_op, z0_out.id, z0_in.id, z0.id);

  auto tile_size = z0_in.size;
  EXPECT_EQ(result_op.attr.sched.axis[0], z0_out.id);
  EXPECT_EQ(result_op.attr.sched.axis[1], z0_in.id);
  EXPECT_EQ(result_op.outputs[0].axis[0], z0_out.id);
  EXPECT_EQ(result_op.outputs[0].axis[1], z0_in.id);
  EXPECT_EQ(result_op.outputs[0].repeats[0], z0.size / tile_size);
  EXPECT_EQ(result_op.outputs[0].repeats[1], tile_size);
  EXPECT_EQ(result_op.outputs[0].strides[0], tile_size);
  EXPECT_EQ(result_op.outputs[0].strides[1], SizeExpr{});
}

TEST(Ascir_AxisOperations, ApplyMerge_OnNode_WillMergeNodeAxis) {
  auto graph = CreateTestGraph();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", ascir::SizeExpr{{s0.id}});
  auto z1 = graph.CreateAxis("z0", ascir::SizeExpr{{s1.id}});

  Data data("test_op");

  data.attr.sched.axis = {z0.id, z1.id};
  data.y.axis = {z0.id, z1.id};
  data.y.repeats = {SizeExpr{{s0.id}}, SizeExpr{{s1.id}}};
  data.y.strides = {SizeExpr{{s1.id}}, SizeExpr{}};
  graph.SetInputs({data});

  auto result_op = graph.Find("test_op");

  auto z_new = graph.MergeAxis({z0.id, z1.id});
  graph.ApplyMerge(result_op, z_new.id, {z0.id, z1.id});

  EXPECT_EQ(result_op.attr.sched.axis[0], z_new.id);
  EXPECT_EQ(result_op.outputs[0].axis[0], z_new.id);
  EXPECT_EQ(result_op.outputs[0].repeats[0], z0.size * z1.size);
  EXPECT_EQ(result_op.outputs[0].strides[0], SizeExpr{});
}

TEST(Ascir_Utils, DebugHintGraphStr_WillShowAxisInfo) {
  Data data("test_op");
  ascir::Graph graph("test_graph");
  graph.SetInputs({data});

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1", 100);
  auto s0_block = graph.CreateSizeVar("s0_block", 100);

  auto z0_out = graph.CreateAxis("z0_out", SizeExpr{{s0.id}, {s0_block.id}});
  auto z0_in = graph.CreateAxis("z0_in", SizeExpr{{s0_block.id}});
  auto z1 = graph.CreateAxis("z1", SizeExpr{{s1.id}});

  data.attr.sched.exec_order = 0;
  data.attr.sched.axis = {z0_out.id, z0_in.id, z1.id};

  data.y.axis = {z0_out.id, z0_in.id, z1.id};
  data.y.repeats = {s0 / s0_block, s0_block, s1};
  data.y.strides = {s0_block * s1, s1, {}};

  auto result_str = ascir::utils::DebugHintGraphStr(graph);

  EXPECT_EQ(result_str, string{"Graph: test_graph\n"
                               "Sizes:\n"
                               "  s0: VAR\n"
                               "  s1: CONST(100)\n"
                               "  s0_block: CONST(100)\n"
                               "Axis:\n"
                               "  z0_out: s0/s0_block, ORIGINAL\n"
                               "  z0_in: s0_block, ORIGINAL\n"
                               "  z1: s1, ORIGINAL\n"
                               "Nodes:\n"
                               "  test_op: Data (0)\n"
                               "    .axis = {z0_out, z0_in, z1, }\n"
                               "    .x = (nil)\n"
                               "    .y.dtype = float32\n"
                               "    .y.axis = {z0_out, z0_in, z1, }\n"
                               "    .y.repeats = {s0/s0_block, s0_block, s1, }\n"
                               "    .y.strides = {s1*s0_block, s1, 1, }\n"
                               "    .y.vectorized_axis = {}\n"});
}

TEST(Ascir_Utils, DebugImplGraphStr) {
  Data data("test_op");
  ascir::Graph graph("test_graph");
  graph.SetInputs({data});

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1", 100);
  auto s0_block = graph.CreateSizeVar("s0_block", 100);

  auto z0_out = graph.CreateAxis("z0_out", SizeExpr{{s0.id}, {s0_block.id}});
  auto z0_in = graph.CreateAxis("z0_in", SizeExpr{{s0_block.id}});
  auto z1 = graph.CreateAxis("z1", SizeExpr{{s1.id}});

  auto data_op = graph.Find(data.GetName().c_str());
  data_op.attr.api.type = ascir::API_TYPE_BUFFER;
  data_op.attr.api.type = ascir::UNIT_NONE;
  data_op.attr.sched.exec_order = 0;
  data_op.attr.sched.axis = {z0_out.id, z0_in.id, z1.id};

  auto data_y = data_op.outputs[0];
  data_y.axis = {z0_out.id, z0_in.id, z1.id};
  data.y.repeats = {s0 / s0_block, s0_block, s1};
  data.y.strides = {s0_block * s1, s1, {}};
  data_y.mem.tensor_id = 0;
  data_y.mem.alloc_type = ALLOC_TYPE_QUEUE;
  data_y.mem.hardware = MEM_HARDWARE_UB;
  data_y.mem.position = POSITION_VECIN;
  data_y.buf.id = ID_NONE;
  data_y.que.id = 0;
  data_y.que.depth = 2;
  data_y.que.buf_num = 2;
  data_y.opt.ref_tensor = ID_NONE;
  data_y.opt.merge_scope = ID_NONE;

  auto result_str = ascir::utils::DebugImplGraphStr(graph);

  EXPECT_EQ(result_str, string{
                            "Graph: test_graph\n"
                            "Sizes:\n"
                            "  s0: VAR\n"
                            "  s1: CONST(100)\n"
                            "  s0_block: CONST(100)\n"
                            "Axis:\n"
                            "  z0_out: s0/s0_block, ORIGINAL\n"
                            "  z0_in: s0_block, ORIGINAL\n"
                            "  z1: s1, ORIGINAL\n"
                            "Nodes:\n"
                            "  test_op: Data (0)\n"
                            "    .axis = {z0_out, z0_in, z1, }\n"
                            "    .api:\n"
                            "      .type = Buffer\n"
                            "      .unit = None\n"
                            "    .x = (nil)\n"
                            "    .y.dtype = float32\n"
                            "    .y.axis = {z0_out, z0_in, z1, }\n"
                            "    .y.repeats = {s0/s0_block, s0_block, s1, }\n"
                            "    .y.strides = {s1*s0_block, s1, 1, }\n"
                            "    .y.vectorized_axis = {}\n"
                            "    .y.mem:\n"
                            "      .tensor_id = 0\n"
                            "      .alloc_type = Queue\n"
                            "      .hardware = UB\n"
                            "      .position = VECIN\n"
                            "    .y.que:\n"
                            "      .id = 0\n"
                            "      .depth = 2\n"
                            "      .buf_num = 2\n"
                            "    .y.opt:\n"
                            "      .ref_tensor = (nil)\n"
                            "      .merge_scope = (nil)\n"
                            });
}
