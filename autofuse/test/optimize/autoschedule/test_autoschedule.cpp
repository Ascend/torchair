#include <ascir.h>
#include <ascir_ops.h>
#include <ascir_utils.h>
#include <iostream>

#include "gtest/gtest.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "graph_utils_ex.h"

#define private public
#include "autoschedule.h"

using namespace std;
using namespace ascir;
using namespace optimize::autoschedule;

static void Construct_LoadAbsStore(ascir::Graph& graph) {
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);

    ops::Data x("x");
    x.attr.sched.exec_order = 0;
    x.attr.sched.axis = {z0.id, z1.id, z2.id};
    x.attr.hint.compute_type = COMPUTE_DATA;
    x.y.dtype = ge::DT_FLOAT16;
    x.y.axis = {z0.id, z1.id, z2.id};
    x.y.repeats = {s0, s1, s2};
    x.y.strides = {s1*s2, s2, {}};

    ops::Load load("load");
    load.x = x;
    load.attr.sched.exec_order = 1;
    load.attr.sched.axis = {z0.id, z1.id, z2.id};
    load.attr.hint.compute_type = COMPUTE_LOAD;
    load.y.dtype = ge::DT_FLOAT16;
    load.y.axis = {z0.id, z1.id, z2.id};
    load.y.repeats = {s0, s1, s2};
    load.y.strides = {s1*s2, s2, {}};

    ops::Abs abs("abs");
    abs.x = load.y;
    abs.attr.sched.exec_order = 2;
    abs.attr.sched.axis = {z0.id, z1.id, z2.id};
    abs.attr.hint.compute_type = COMPUTE_ELEWISE;
    abs.y.dtype = ge::DT_FLOAT16;
    abs.y.axis = {z0.id, z1.id, z2.id};
    abs.y.repeats = {s0, s1, s2};
    abs.y.strides = {s1*s2, s2, {}};

    ops::Store store("store");
    store.x = abs.y;
    store.attr.sched.exec_order = 3;
    store.attr.sched.axis = {z0.id, z1.id, z2.id};
    store.attr.hint.compute_type = COMPUTE_STORE;
    store.y.dtype = ge::DT_FLOAT16;
    store.y.axis = {z0.id, z1.id, z2.id};
    store.y.repeats = {s0, s1, s2};
    store.y.strides = {s1*s2, s2, {}};

    ops::Data y("y");
    y.x = store.y;
    y.attr.sched.exec_order = 4;
    y.attr.sched.axis = {z0.id, z1.id, z2.id};
    y.attr.hint.compute_type = COMPUTE_DATA;
    y.y.dtype = ge::DT_FLOAT16;
    y.y.axis = {z0.id, z1.id, z2.id};
    y.y.repeats = {s0, s1, s2};
    y.y.strides = {s1*s2, s2, {}};

    graph.SetInputs({x});
    graph.SetOutputs({y});
}

static void Construct_ElementwiseAbs(ascir::Graph& graph) {
    auto s0 = graph.CreateSizeVar("s0");

    auto z0 = graph.CreateAxis("z0", s0);

    auto one = ascir::SizeExpr::One();
    ops::Data x("x");
    x.attr.sched.exec_order = 0;
    x.attr.sched.axis = {z0.id};
    x.attr.hint.compute_type = COMPUTE_DATA;
    x.y.dtype = ge::DT_FLOAT16;
    x.y.axis = {z0.id};
    x.y.repeats = {s0};
    x.y.strides = {one};

    ops::Load load("load");
    load.x = x;
    load.attr.sched.exec_order = 1;
    load.attr.sched.axis = {z0.id};
    load.attr.hint.compute_type = COMPUTE_LOAD;
    load.y.dtype = ge::DT_FLOAT16;
    load.y.axis = {z0.id};
    load.y.repeats = {s0};
    load.y.strides = {one};

    ops::Abs abs("abs");
    abs.x = load.y;
    abs.attr.sched.exec_order = 2;
    abs.attr.sched.axis = {z0.id};
    abs.attr.hint.compute_type = COMPUTE_ELEWISE;
    abs.y.dtype = ge::DT_FLOAT16;
    abs.y.axis = {z0.id};
    abs.y.repeats = {s0};
    abs.y.strides = {one};

    ops::Store store("store");
    store.x = abs.y;
    store.attr.sched.exec_order = 3;
    store.attr.sched.axis = {z0.id};
    store.attr.hint.compute_type = COMPUTE_STORE;
    store.y.dtype = ge::DT_FLOAT16;
    store.y.axis = {z0.id};
    store.y.repeats = {s0};
    store.y.strides = {one};

    ops::Data y("y");
    y.x = store.y;
    y.attr.sched.exec_order = 4;
    y.attr.sched.axis = {z0.id};
    y.attr.hint.compute_type = COMPUTE_DATA;
    y.y.dtype = ge::DT_FLOAT16;
    y.y.axis = {z0.id};
    y.y.repeats = {s0};
    y.y.strides = {one};

    graph.SetInputs({x});
    graph.SetOutputs({y});
}

static void Construct_ElementwiseFusion(ascir::Graph& graph) {
    auto s0 = graph.CreateSizeVar("s0");

    auto z0 = graph.CreateAxis("z0", s0);

    auto one = ascir::SizeExpr::One();
    ops::Data x("x");
    x.attr.sched.exec_order = 0;
    x.attr.sched.axis = {z0.id};
    x.attr.hint.compute_type = COMPUTE_DATA;
    x.y.dtype = ge::DT_FLOAT16;
    x.y.axis = {z0.id};
    x.y.repeats = {s0};
    x.y.strides = {one};

    ops::Load load("load");
    load.x = x;
    load.attr.sched.exec_order = 1;
    load.attr.sched.axis = {z0.id};
    load.attr.hint.compute_type = COMPUTE_LOAD;
    load.y.dtype = ge::DT_FLOAT16;
    load.y.axis = {z0.id};
    load.y.repeats = {s0};
    load.y.strides = {one};

    ops::Abs abs0("abs0");
    abs0.x = load.y;
    abs0.attr.sched.exec_order = 2;
    abs0.attr.sched.axis = {z0.id};
    abs0.attr.hint.compute_type = COMPUTE_ELEWISE;
    abs0.y.dtype = ge::DT_FLOAT16;
    abs0.y.axis = {z0.id};
    abs0.y.repeats = {s0};
    abs0.y.strides = {one};

    ops::Abs abs1("abs1");
    abs1.x = abs0.y;
    abs1.attr.sched.exec_order = 3;
    abs1.attr.sched.axis = {z0.id};
    abs1.attr.hint.compute_type = COMPUTE_ELEWISE;
    abs1.y.dtype = ge::DT_FLOAT16;
    abs1.y.axis = {z0.id};
    abs1.y.repeats = {s0};
    abs1.y.strides = {one};

    ops::Store store("store");
    store.x = abs1.y;
    store.attr.sched.exec_order = 4;
    store.attr.sched.axis = {z0.id};
    store.attr.hint.compute_type = COMPUTE_STORE;
    store.y.dtype = ge::DT_FLOAT16;
    store.y.axis = {z0.id};
    store.y.repeats = {s0};
    store.y.strides = {one};

    ops::Data y("y");
    y.x = store.y;
    y.attr.sched.exec_order = 5;
    y.attr.sched.axis = {z0.id};
    y.attr.hint.compute_type = COMPUTE_DATA;
    y.y.dtype = ge::DT_FLOAT16;
    y.y.axis = {z0.id};
    y.y.repeats = {s0};
    y.y.strides = {one};

    graph.SetInputs({x});
    graph.SetOutputs({y});
}

TEST(E2E_AutoScheduleAbs, TilingGroup_gen_elementwise_tilingGroup) {
  ascir::Graph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  auto abs = graph.Find("abs");

  TilingGroup group;
  group.GenElewiseTilingGroup(abs);

  std::vector<AxisId> eg;
  for (auto axis : abs.attr.sched.axis()) {
    eg.push_back(axis);
  }
  TilingGroup expect_group(eg);

  EXPECT_EQ(group, expect_group);
}

TEST(E2E_AutoScheduleAbs, TilingGroup_compare) {
  std::vector<AxisId> yGroup0 = {0, 1, 3};
  std::vector<AxisId> yGroup1 = {0, 1, 3};
  TilingGroup group0(yGroup0);
  TilingGroup group1(yGroup0);
  EXPECT_EQ(group0, group1);
}

TEST(E2E_AutoScheduleAbs, AutoSchedule_eletwise_gen_tilingGroup) {
  ascir::Graph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  auto abs = graph.Find("abs");

  std::vector<ImplGraph> impl_graphs;
  AutoSchedule autoSchedule(graph, impl_graphs);
  autoSchedule.GenAxesGroup();
  EXPECT_EQ(autoSchedule.axes_group_.XGroup().size(), 1);
  EXPECT_EQ(autoSchedule.axes_group_.YGroup().size(), 3);
  EXPECT_EQ(autoSchedule.axes_group_.RGroup().size(), 1);

  ascir::AxisId default_id = -1;
  std::vector<ascir::AxisId> x_group = {default_id};
  std::vector<ascir::AxisId> y_group(abs.attr.sched.axis());
  std::vector<ascir::AxisId> r_group = {default_id};
  EXPECT_EQ(autoSchedule.axes_group_.XGroup(), x_group);
  EXPECT_EQ(autoSchedule.axes_group_.YGroup(), y_group);
  EXPECT_EQ(autoSchedule.axes_group_.RGroup(), r_group);
}

TEST(E2E_AutoScheduleAbs, AutoSchedule_eletwise_fusion_gen_tilingGroup) {
  ascir::Graph graph("LoadAbsStore");
  Construct_ElementwiseFusion(graph);

  auto abs = graph.Find("abs1");

  std::vector<ImplGraph> impl_graphs;
  AutoSchedule autoSchedule(graph, impl_graphs);
  autoSchedule.GenAxesGroup();
  EXPECT_EQ(autoSchedule.axes_group_.XGroup().size(), 1);
  EXPECT_EQ(autoSchedule.axes_group_.YGroup().size(), 1);
  EXPECT_EQ(autoSchedule.axes_group_.RGroup().size(), 1);

  ascir::AxisId default_id = -1;
  std::vector<ascir::AxisId> x_group = {default_id};
  std::vector<ascir::AxisId> y_group(abs.attr.sched.axis());
  std::vector<ascir::AxisId> r_group = {default_id};
  EXPECT_EQ(autoSchedule.axes_group_.XGroup(), x_group);
  EXPECT_EQ(autoSchedule.axes_group_.YGroup(), y_group);
  EXPECT_EQ(autoSchedule.axes_group_.RGroup(), r_group);
}

TEST(E2E_AutoScheduleAbs, AutoSchedule_node_group) {
  ascir::Graph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  auto abs = graph.Find("abs");
  ascir::SchAttr sch_attr{abs.outputs[0].desc};
  sch_attr.group_id = 1;

  auto abs_copy = graph.Find("abs");
  ascir::SchAttr sch_attr_copy{abs_copy.outputs[0].desc};
  auto expect_group_id = sch_attr_copy.group_id;

  EXPECT_EQ(expect_group_id, 1);
}

TEST(E2E_AutoScheduleAbs, AutoSchedule_Tiling) {
  ascir::Graph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);
  
  auto abs = graph.Find("abs");
  std::vector<ascir::AxisId> y_group = abs.attr.sched.axis;

  TilingGroup axes_group(y_group);
  axes_group.axes_order_ = {0, 1, 2};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = y_group[1];
  tiling_case.block_tiling_id = 0;
  Scheduler scheduler(graph, axes_group, tiling_case);
  scheduler.Tiling();
  
  EXPECT_EQ(scheduler.tiling_case_.ub_tiling_id_x, -1);
  EXPECT_EQ(scheduler.tiling_case_.ub_tiling_id_y, 1);
  EXPECT_EQ(scheduler.tiling_case_.ub_tiling_id_r, -1);
  EXPECT_EQ(scheduler.tiling_case_.block_tiling_id, 5);
  EXPECT_EQ(graph.axis().size(), 8);
  EXPECT_EQ(std::get<0>(scheduler.tiling_case_.ub_tiling_y).id, 3);
  EXPECT_EQ(std::get<1>(scheduler.tiling_case_.ub_tiling_y).id, 4);
  EXPECT_EQ(std::get<0>(scheduler.tiling_case_.block_tling).id, 6);
  EXPECT_EQ(std::get<1>(scheduler.tiling_case_.block_tling).id, 7);
}

TEST(E2E_AutoScheduleAbs, Autoschedule_scheduler_elementwise_3axis)
{
    ascir::Graph graph("LoadAbsStore");
    Construct_LoadAbsStore(graph);

    ascir::Graph except_graph("LoadAbsStore");
    except_graph.CopyFrom(graph);

    auto abs = graph.Find("abs");
    std::vector<ascir::AxisId> y_group = abs.attr.sched.axis;

    TilingGroup axes_group(y_group);
    axes_group.axes_order_ = {0, 1, 2};
    TilingCase tiling_case;
    tiling_case.ub_tiling_id_y = 1;
    tiling_case.block_tiling_id = 0;
    Scheduler scheduler(graph, axes_group, tiling_case);
    scheduler.DoScheduler();
    auto dump_graph = ascir::utils::DebugStr(graph);

    int block_axis_id = 0;
    int ub_axis_id = 1;
    
    auto output = except_graph.Find("store");
    // split ub
    auto z_ub_id = output.attr.sched.axis[ub_axis_id];
    auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);

    // fuse outer axes
    std::vector<AxisId> axes{output.attr.sched.axis[0], z_ub_out.id};
    auto block_axis = except_graph.MergeAxis(axes);
    
    // split block
    auto [z_block_out, z_block_in] = except_graph.BlockSplit(block_axis.id);

    vector<AxisId> vectorize_axis = {z_ub_in.id, output.attr.sched.axis[2]};

    for (auto n : except_graph.GetAllNodes()) {
      if (n.attr.hint.compute_type == COMPUTE_DATA) {
        continue;
      }
      except_graph.ApplySplit(n, z_ub_out.id, z_ub_in.id, z_ub_id);
      except_graph.ApplyMerge(n, block_axis.id);
      except_graph.ApplySplit(n, z_block_out.id, z_block_in.id, block_axis.id);
      n.outputs[0].vectorized_axis = vectorize_axis;
    }

    auto dump_except_graph = ascir::utils::DebugStr(except_graph);

    EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST(E2E_AutoScheduleAbs, Autoschedule_scheduler_elementwise_1axis)
{
    ascir::Graph graph("LoadAbsStore");
    Construct_ElementwiseAbs(graph);

    ascir::Graph except_graph("LoadAbsStore");
    except_graph.CopyFrom(graph);

    auto abs = graph.Find("abs");
    std::vector<ascir::AxisId> y_group = abs.attr.sched.axis;

    TilingGroup axes_group(y_group);
    axes_group.axes_order_ = {0};
    TilingCase tiling_case;
    tiling_case.ub_tiling_id_y = 0;
    tiling_case.block_tiling_id = 0;
    Scheduler scheduler(graph, axes_group, tiling_case);
    scheduler.DoScheduler();
    auto dump_graph = ascir::utils::DebugStr(graph);

    int block_axis_id = 0;
    int ub_axis_id = 0;
    
    auto output = except_graph.Find("store");
    // split ub
    auto z_ub_id = output.attr.sched.axis[ub_axis_id];
    auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);
    
    // split block
    auto [z_block_out, z_block_in] = except_graph.BlockSplit(z_ub_out.id);

    vector<AxisId> vectorize_axis = {z_ub_in.id};

    for (auto n : except_graph.GetAllNodes()) {
      if (n.attr.hint.compute_type == COMPUTE_DATA) {
        continue;
      }
      except_graph.ApplySplit(n, z_ub_out.id, z_ub_in.id, z_ub_id);
      except_graph.ApplySplit(n, z_block_out.id, z_block_in.id, z_ub_out.id);
      n.outputs[0].vectorized_axis = vectorize_axis;
    }
    
    auto dump_except_graph = ascir::utils::DebugStr(except_graph);

    EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST(E2E_AutoScheduleAbs, Autoschedule_autoschedule_elementwise_1axis)
{
    ascir::Graph graph("LoadAbsStore");
    Construct_ElementwiseAbs(graph);

    ascir::Graph except_graph("LoadAbsStore_general_0_nil_0_nil");
    except_graph.CopyFrom(graph);

    std::vector<ImplGraph> impl_graphs;
    AutoSchedule autoschedule(graph, impl_graphs);
    autoschedule.DoAutoSchedule();
    EXPECT_EQ(impl_graphs.size(), 1);
    auto dump_graph = ascir::utils::DebugStr(impl_graphs[0]);

    int block_axis_id = 0;
    int ub_axis_id = 0;
    
    auto output = except_graph.Find("store");
    // split ub
    auto z_ub_id = output.attr.sched.axis[ub_axis_id];
    auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);
    
    // split block
    auto [z_block_out, z_block_in] = except_graph.BlockSplit(z_ub_out.id);

    vector<AxisId> vectorize_axis = {z_ub_in.id};

    for (auto n : except_graph.GetAllNodes()) {
      if (n.attr.hint.compute_type == COMPUTE_DATA) {
        continue;
      }
      except_graph.ApplySplit(n, z_ub_out.id, z_ub_in.id, z_ub_id);
      except_graph.ApplySplit(n, z_block_out.id, z_block_in.id, z_ub_out.id);
      n.outputs[0].vectorized_axis = vectorize_axis;
    }
    
    auto dump_except_graph = ascir::utils::DebugStr(except_graph);

    EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST(E2E_AutoScheduleAbs, Autoschedule_autoschedule_elementwise_fusion)
{
    ascir::Graph graph("AbsFusion");
    Construct_ElementwiseFusion(graph);

    ascir::Graph except_graph("AbsFusion_general_0_nil_0_nil");
    except_graph.CopyFrom(graph);

    std::vector<ImplGraph> impl_graphs;
    AutoSchedule autoschedule(graph, impl_graphs);
    autoschedule.DoAutoSchedule();
    EXPECT_EQ(impl_graphs.size(), 1);
    auto dump_graph = ascir::utils::DebugStr(impl_graphs[0]);

    int block_axis_id = 0;
    int ub_axis_id = 0;
    
    auto output = except_graph.Find("store");
    // split ub
    auto z_ub_id = output.attr.sched.axis[ub_axis_id];
    auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);
    
    // split block
    auto [z_block_out, z_block_in] = except_graph.BlockSplit(z_ub_out.id);

    vector<AxisId> vectorize_axis = {z_ub_in.id};

    for (auto n : except_graph.GetAllNodes()) {
      if (n.attr.hint.compute_type == COMPUTE_DATA) {
        continue;
      }
      except_graph.ApplySplit(n, z_ub_out.id, z_ub_in.id, z_ub_id);
      except_graph.ApplySplit(n, z_block_out.id, z_block_in.id, z_ub_out.id);
      n.outputs[0].vectorized_axis = vectorize_axis;
    }
    
    auto dump_except_graph = ascir::utils::DebugStr(except_graph);

    EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST(E2E_AutoScheduleAbs, Autoschedule_number_of_node)
{
    ascir::Graph graph("LoadAbsStore");
    ops::Data x("x");
    x.attr.sched.exec_order = 0;
    x.attr.hint.compute_type = COMPUTE_DATA;

    ops::Load load("load");
    load.x = x;
    load.attr.sched.exec_order = 1;
    load.attr.hint.compute_type = COMPUTE_LOAD;

    ops::Abs abs0("abs0");
    abs0.x = load.y;
    abs0.attr.sched.exec_order = 2;
    abs0.attr.hint.compute_type = COMPUTE_ELEWISE;

    ops::Abs abs1("abs1");
    abs1.x = abs0.y;
    abs1.attr.sched.exec_order = 3;
    abs1.attr.hint.compute_type = COMPUTE_ELEWISE;

    ops::Abs abs2("abs2");
    abs2.x = abs1.y;
    abs2.attr.sched.exec_order = 4;
    abs2.attr.hint.compute_type = COMPUTE_ELEWISE;

    ops::Store store("store");
    store.x = abs2.y;
    store.attr.sched.exec_order = 5;
    store.attr.hint.compute_type = COMPUTE_STORE;

    ops::Data y("y");
    y.x = store.y;
    y.attr.sched.exec_order = 7;
    y.attr.hint.compute_type = COMPUTE_DATA;

    graph.SetInputs({x});
    graph.SetOutputs({y});

    TilingGroup axes_group;
    TilingCase tiling_case;
    Scheduler autoSchedule(graph, axes_group, tiling_case);
    autoSchedule.NodeNumber();
    
    auto load_result = graph.Find(load.GetName().c_str());
    EXPECT_EQ(load_result.outputs[0].opt.reuse_id, 0);
    auto abs0_result = graph.Find(abs0.GetName().c_str());
    EXPECT_EQ(abs0_result.outputs[0].opt.reuse_id, 1);
    auto abs1_result = graph.Find(abs1.GetName().c_str());
    EXPECT_EQ(abs1_result.outputs[0].opt.reuse_id, 0);
    auto abs2_result = graph.Find(abs2.GetName().c_str());
    EXPECT_EQ(abs2_result.outputs[0].opt.reuse_id, 1);
}