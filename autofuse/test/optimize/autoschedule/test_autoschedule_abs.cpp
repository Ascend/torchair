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

    auto z0 = graph.CreateAxis("z0", {{s0.id}});
    auto z1 = graph.CreateAxis("z1", {{s1.id}});
    auto z2 = graph.CreateAxis("z2", {{s2.id}});

    ops::Data x("x");
    x.attr.sched.exec_order = 0;
    x.attr.sched.axis = {z0.id, z1.id, z2.id};
    x.attr.hint.compute_type = COMPUTE_DATA;
    x.y.dtype = ge::DT_FLOAT16;
    x.y.axis = {z0.id, z1.id, z2.id};
    x.y.repeats = {{{s0.id}}, {{s1.id}}, {{s2.id}}};
    x.y.strides = {{{s1.id, s2.id}}, {{s2.id}}, {{}}};

    ops::Load load("load");
    load.x = x;
    load.attr.sched.exec_order = 1;
    load.attr.sched.axis = {z0.id, z1.id, z2.id};
    load.attr.hint.compute_type = COMPUTE_LOAD;
    load.y.dtype = ge::DT_FLOAT16;
    load.y.axis = {z0.id, z1.id, z2.id};
    load.y.repeats = {{{s0.id}}, {{s1.id}}, {{s2.id}}};
    load.y.strides = {{{s1.id, s2.id}}, {{s2.id}}, {{}}};

    ops::Abs abs("abs");
    abs.x = load.y;
    abs.attr.sched.exec_order = 2;
    abs.attr.sched.axis = {z0.id, z1.id, z2.id};
    abs.attr.hint.compute_type = COMPUTE_ELEWISE;
    abs.y.dtype = ge::DT_FLOAT16;
    abs.y.axis = {z0.id, z1.id, z2.id};
    abs.y.repeats = {{{s0.id}}, {{s1.id}}, {{s2.id}}};
    abs.y.strides = {{{s1.id, s2.id}}, {{s2.id}}, {{}}};

    ops::Store store("store");
    store.x = abs.y;
    store.attr.sched.exec_order = 3;
    store.attr.sched.axis = {z0.id, z1.id, z2.id};
    store.attr.hint.compute_type = COMPUTE_STORE;
    store.y.dtype = ge::DT_FLOAT16;
    store.y.axis = {z0.id, z1.id, z2.id};
    store.y.repeats = {{{s0.id}}, {{s1.id}}, {{s2.id}}};
    store.y.strides = {{{s1.id, s2.id}}, {{s2.id}}, {{}}};

    ops::Data y("y");
    y.x = store.y;
    y.attr.sched.exec_order = 4;
    y.attr.sched.axis = {z0.id, z1.id, z2.id};
    y.attr.hint.compute_type = COMPUTE_DATA;
    y.y.dtype = ge::DT_FLOAT16;
    y.y.axis = {z0.id, z1.id, z2.id};
    y.y.repeats = {{{s0.id}}, {{s1.id}}, {{s2.id}}};
    y.y.strides = {{{s1.id, s2.id}}, {{s2.id}}, {{}}};

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

TEST(E2E_AutoScheduleAbs, AutoSchedule_gen_tilingGroup) {
  ascir::Graph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  auto abs = graph.Find("abs");

  AutoSchedule autoSchedule(graph);
  autoSchedule.GenTilingGroup();
  EXPECT_EQ(autoSchedule.tilingGroup_.size(), 1);

  std::vector<AxisId> eg;
  for (auto axis : abs.attr.sched.axis()) {
    eg.push_back(axis);
  }
  TilingGroup expect_group(eg);
  EXPECT_EQ(autoSchedule.tilingGroup_.count(expect_group), 1);
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
  
  AutoSchedule autoSchedule(graph);
  int ub_tiling_id = 1;
  int block_tiling_id = 0;
  autoSchedule.tilingCase_.ubTilingIdY = ub_tiling_id;
  autoSchedule.tilingCase_.blockTilingId = block_tiling_id;
  autoSchedule.Tiling();
  
  EXPECT_EQ(autoSchedule.tilingCase_.ubTilingIdX, -1);
  EXPECT_EQ(autoSchedule.tilingCase_.ubTilingIdY, 1);
  EXPECT_EQ(autoSchedule.tilingCase_.ubTilingIdR, -1);
  EXPECT_EQ(autoSchedule.tilingCase_.blockTilingId, 0);
  EXPECT_EQ(graph.axis().size(), 7);
  EXPECT_EQ(std::get<0>(autoSchedule.tilingCase_.ubTilingY).id, 3);
  EXPECT_EQ(std::get<1>(autoSchedule.tilingCase_.ubTilingY).id, 4);
  EXPECT_EQ(std::get<0>(autoSchedule.tilingCase_.blockTiling).id, 5);
  EXPECT_EQ(std::get<1>(autoSchedule.tilingCase_.blockTiling).id, 6);
}

TEST(E2E_AutoScheduleAbs, Autoschedule_autoschedule)
{
    ascir::Graph graph("LoadAbsStore");
    Construct_LoadAbsStore(graph);

    AutoSchedule autoSchedule(graph);
    autoSchedule.Scheduler();
    auto dump_graph = ascir::utils::DebugStr(graph);

    ascir::Graph except_graph("LoadAbsStore");
    Construct_LoadAbsStore(except_graph);

    int block_axis_id = 0;
    int ub_axis_id = 1;
    
    auto output = except_graph.Find("store");
    // split ub
    auto z_ub_id = output.attr.sched.axis[ub_axis_id];
    auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);
    
    // split block
    auto z_block_id = output.attr.sched.axis[block_axis_id];
    auto [z_block_out, z_block_in] = except_graph.BlockSplit(z_block_id);

    // find vectorize axis
    vector<AxisId> vectorize_axis = {z_ub_in.id};
    bool find_tile_in = false;
    for (auto& axis: output.outputs[0].axis()) {
      if (axis == z_ub_id) {
        find_tile_in = true;
        continue;
      }

      if (find_tile_in) {
        vectorize_axis.push_back(axis);
      }
    }

    for (auto n : except_graph.GetAllNodes()) {
      cout << n->GetName() << ": " << n.attr.sched.exec_order << endl;
      except_graph.ApplySplit(n, z_ub_out.id, z_ub_in.id, z_ub_id);
      except_graph.ApplySplit(n, z_block_out.id, z_block_in.id, z_block_id);

      if (n.attr.hint.compute_type != COMPUTE_DATA) {
        n.outputs[0].vectorized_axis = vectorize_axis;
      }
    }

    auto dump_except_graph = ascir::utils::DebugStr(except_graph);

    EXPECT_EQ(dump_graph, dump_except_graph);
}
