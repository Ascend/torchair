#include <ascir.h>
#include <ascir_ops.h>
#include <ascir_utils.h>
#include <iostream>

#include "gtest/gtest.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "optimize.h"
#include "e2e_softmax.h"

#include "graph_utils_ex.h"

#define private public
#include "autoschedule.h"

using namespace std;
using namespace ascir;
using namespace optimize::autoschedule;
using namespace ascir::ops;

void Construct_Softmax(ascir::HintGraph &graph) {
  auto ONE = SizeExpr::One();
  auto ZERO = SizeExpr::Zero();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0 * s1 * s2);
  auto z1 = graph.CreateAxis("z1", s3);

  auto axis = {z0.id, z1.id};

  int exec_order = 0;

  Data arg4_1("arg4_1");
  arg4_1.attr.sched.exec_order = exec_order++;
  arg4_1.attr.hint.compute_type = COMPUTE_DATA;
  arg4_1.y.dtype = ge::DT_FLOAT16;

  Load b0_load("b0_load");
  b0_load.x = arg4_1;
  b0_load.attr.sched.exec_order = exec_order++;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.hint.compute_type = COMPUTE_LOAD;
  b0_load.y.dtype = ge::DT_FLOAT16;
  b0_load.y.axis = axis;
  b0_load.y.repeats = {s0*s1*s2, s3};
  b0_load.y.strides = {s3, ONE};

  Max b0_max("b0_max");
  b0_max.x = b0_load.y;
  b0_max.attr.sched.exec_order = exec_order++;
  b0_max.attr.sched.axis = axis;
  b0_max.attr.hint.compute_type = COMPUTE_REDUCE;
  b0_max.y.dtype = ge::DT_FLOAT16;
  b0_max.y.axis = axis;
  b0_max.y.repeats = {s0*s1*s2, s3};
  b0_max.y.strides = {ONE, ZERO};

  Data buf1("buf1");
  Load b1_load("b1_load");
  b1_load.x = arg4_1;
  b1_load.attr.sched.exec_order = exec_order++;
  b1_load.attr.sched.axis = axis;
  b1_load.attr.hint.compute_type = COMPUTE_LOAD;
  b1_load.y.dtype = ge::DT_FLOAT16;
  b1_load.y.axis = axis;
  b1_load.y.repeats = {s0*s1*s2, s3};
  b1_load.y.strides = {s3, ONE};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b0_max.y;
  b1_broadcast.attr.sched.exec_order = exec_order++;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.hint.compute_type = COMPUTE_BROADCAST;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  b1_broadcast.y.axis = axis;
  b1_broadcast.y.repeats = {s0*s1*s2, s3};
  b1_broadcast.y.strides = {s3, ONE};

  Sub b1_sub("b1_sub");
  b1_sub.x1 = b1_load.y;
  b1_sub.x2 = b1_broadcast.y;
  b1_sub.attr.sched.exec_order = exec_order++;
  b1_sub.attr.sched.axis = axis;
  b1_sub.attr.hint.compute_type = COMPUTE_ELEWISE;
  b1_sub.y.dtype = ge::DT_FLOAT16;
  b1_sub.y.axis = axis;
  b1_sub.y.repeats = {s0*s1*s2, s3};
  b1_sub.y.strides = {s3, ONE};

  Exp b1_exp("b1_exp");
  b1_exp.x = b1_sub.y;
  b1_exp.attr.sched.exec_order = exec_order++;
  b1_exp.attr.sched.axis = axis;
  b1_exp.attr.hint.compute_type = COMPUTE_ELEWISE;
  b1_exp.y.dtype = ge::DT_FLOAT16;
  b1_exp.y.axis = axis;
  b1_exp.y.repeats = {s0*s1*s2, s3};
  b1_exp.y.strides = {s3, ONE};

  Store b1_store("b1_store");
  b1_store.x = b1_exp.y;
  b1_store.attr.sched.exec_order = exec_order++;
  b1_store.attr.sched.axis = axis;
  b1_store.attr.hint.compute_type = COMPUTE_STORE;
  b1_store.y.dtype = ge::DT_FLOAT16;
  b1_store.y.axis = axis;
  b1_store.y.repeats = {s0*s1*s2, s3};
  b1_store.y.strides = {s3, ONE};

  buf1.x = b1_store.y;
  buf1.attr.sched.exec_order = exec_order++;
  buf1.attr.hint.compute_type = COMPUTE_WORKSPACE;
  buf1.y.dtype = ge::DT_FLOAT16;

  Load b2_load("b2_load");
  b2_load.x = buf1;
  b2_load.attr.sched.exec_order = exec_order++;
  b2_load.attr.sched.axis = axis;
  b2_load.attr.hint.compute_type = COMPUTE_LOAD;
  b2_load.y.dtype = ge::DT_FLOAT16;
  b2_load.y.axis = axis;
  b2_load.y.repeats = {s0*s1*s2, s3};
  b2_load.y.strides = {s3, ONE};

  Sum b2_sum("b2_sum");
  b2_sum.x = b2_load.y;
  b2_sum.attr.sched.exec_order = exec_order++;
  b2_sum.attr.sched.axis = axis;
  b2_sum.attr.hint.compute_type = COMPUTE_REDUCE;
  b2_sum.y.dtype = ge::DT_FLOAT16;
  b2_sum.y.axis = axis;
  b2_sum.y.repeats = {s0*s1*s2, s3};
  b2_sum.y.strides = {ONE, ZERO};

  Data buf3("buf3");
  Load b3_load("b3_load");
  b3_load.x = buf1;
  b3_load.attr.sched.exec_order = exec_order++;
  b3_load.attr.sched.axis = axis;
  b3_load.attr.hint.compute_type = COMPUTE_LOAD;
  b3_load.y.dtype = ge::DT_FLOAT16;
  b3_load.y.axis = axis;
  b3_load.y.repeats = {s0*s1*s2, s3};
  b3_load.y.strides = {s3, ONE};

  Broadcast b3_broadcast("b3_broadcast");
  b3_broadcast.x = b2_sum.y;
  b3_broadcast.attr.sched.exec_order = exec_order++;
  b3_broadcast.attr.sched.axis = axis;
  b3_broadcast.attr.hint.compute_type = COMPUTE_BROADCAST;
  b3_broadcast.y.dtype = ge::DT_FLOAT16;
  b3_broadcast.y.axis = axis;
  b3_broadcast.y.repeats = {s0 * s1 * s2, s3};
  b3_broadcast.y.strides = {s3, ONE};

  Div b3_div("b3_div");
  b3_div.x1 = b3_load.y;
  b3_div.x2 = b3_broadcast.y;
  b3_div.attr.sched.exec_order = exec_order++;
  b3_div.attr.sched.axis = axis;
  b3_div.attr.hint.compute_type = COMPUTE_ELEWISE;
  b3_div.y.dtype = ge::DT_FLOAT16;
  b3_div.y.axis = axis;
  b3_div.y.repeats = {s0*s1*s2, s3};
  b3_div.y.strides = {s3, ONE};

  Store b3_store("b3_store");
  b3_store.x = b3_div.y;
  b3_store.attr.sched.exec_order = exec_order++;
  b3_store.attr.sched.axis = axis;
  b3_store.attr.hint.compute_type = COMPUTE_STORE;
  b3_store.y.dtype = ge::DT_FLOAT16;
  b3_store.y.axis = axis;
  b3_store.y.repeats = {s0*s1*s2, s3};
  b3_store.y.strides = {s3, ONE};

  buf3.x = b3_store.y;
  buf3.attr.sched.exec_order = exec_order++;
  buf3.attr.hint.compute_type = COMPUTE_DATA;
  buf3.y.dtype = ge::DT_FLOAT16;

  graph.SetInputs({arg4_1});
  graph.SetOutputs({buf3});
}

TEST(E2E_AutoScheduleSoftmax, Autoschedule_autoschedule_softmax_fusion_axesgroup)
{
    ascir::Graph graph("SoftmaxFusion");
    Construct_Softmax(graph);

    ascir::Graph except_graph("SoftmaxFusion_general_0_-1_0_-1");
    except_graph.CopyFrom(graph);

    auto store = graph.Find("b3_store");
    std::vector<ascir::AxisId> axes = store.attr.sched.axis;
    std::vector<ascir::AxisId> y_group = {axes[0]};
    std::vector<ascir::AxisId> r_group = {axes[1]};

    std::vector<ImplGraph> impl_graphs;
    AutoSchedule autoschedule(graph, impl_graphs);
    autoschedule.GenAxesGroup();
    EXPECT_EQ(autoschedule.axes_group_.XGroup().size(), 1);
    EXPECT_EQ(autoschedule.axes_group_.XGroup()[0], -1);
    EXPECT_EQ(autoschedule.axes_group_.YGroup().size(), 1);
    EXPECT_EQ(autoschedule.axes_group_.YGroup(), y_group);
    EXPECT_EQ(autoschedule.axes_group_.RGroup().size(), 1);
    EXPECT_EQ(autoschedule.axes_group_.RGroup(), r_group);
}

TEST(E2E_AutoScheduleSoftmax, Autoschedule_autoschedule_softmax_fusion_tilingcase)
{
    ascir::Graph graph("SoftmaxFusion");
    Construct_Softmax(graph);

    ascir::Graph except_graph("SoftmaxFusion_general_0_-1_0_-1");
    except_graph.CopyFrom(graph);

    auto store = graph.Find("b3_store");
    std::vector<ascir::AxisId> axes = store.attr.sched.axis;

    std::vector<ImplGraph> impl_graphs;
    AutoSchedule autoschedule(graph, impl_graphs);
    autoschedule.GenAxesGroup();
    std::vector<TilingCase> tiling_cases;
    autoschedule.GenTilingCase(tiling_cases);
    EXPECT_EQ(tiling_cases.size(), 1);
    EXPECT_EQ(tiling_cases[0].ub_tiling_id_x, -1);
    EXPECT_EQ(tiling_cases[0].ub_tiling_id_y, axes[0]);
    EXPECT_EQ(tiling_cases[0].ub_tiling_id_r, axes[1]);
    EXPECT_EQ(tiling_cases[0].block_tiling_id, 0);
    EXPECT_EQ(tiling_cases[0].reduce_is_block, false);
}

TEST(E2E_AutoScheduleSoftmax, Autoschedule_autoschedule_softmax_fusion)
{
    ascir::Graph graph("SoftmaxFusion");
    Construct_Softmax(graph);

    ascir::Graph except_graph("SoftmaxFusion_general_0_nui_0_nui");
    except_graph.CopyFrom(graph);

    std::vector<ImplGraph> impl_graphs;
    AutoSchedule autoschedule(graph, impl_graphs);
    autoschedule.DoAutoSchedule();
    EXPECT_EQ(impl_graphs.size(), 1);
    auto dump_graph = ascir::utils::DebugStr(impl_graphs[0]);
    std::cout << dump_graph << std::endl;
}
