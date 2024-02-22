#include "e2e_broadcast.h"

#include "ascir.h"
#include "ascir_ops.h"

using namespace ascir;
using namespace ascir::ops;

void LoadBroadcastStore_BeforeAutofuse(ascir::HintGraph &graph, bool is_f16) {
  auto ONE = SizeExpr::One();
  auto ZERO = SizeExpr::Zero();

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  int exec_order = 0;

  Data x("x");
  x.attr.sched.exec_order = exec_order++;
  if (is_f16) {
    x.y.dtype = ge::DT_FLOAT16;
  } else {
    x.y.dtype = ge::DT_FLOAT;
  }

  Load load("load");
  load.x = x;
  load.attr.sched.exec_order = exec_order++;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.axis = {z0.id, z1.id};
  load.y.repeats = {s0, ONE};
  load.y.strides = {ONE, ZERO};

  Broadcast broadcast("broadcast");
  broadcast.x = load.y;
  broadcast.attr.sched.exec_order = exec_order++;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.repeats = {s0, s1};
  broadcast.y.strides = {s1, ONE};

  Store store("store");
  store.x = broadcast.y;
  store.attr.sched.exec_order = exec_order++;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.axis = {z0.id, z1.id};
  store.y.repeats = {s0, s1};
  store.y.strides = {s1, ONE};

  Output y("y");
  y.x = store.y;
  if (is_f16) {
    y.y.dtype = ge::DT_FLOAT16;
  } else {
    y.y.dtype = ge::DT_FLOAT;
  }

  graph.SetInputs({x});
  graph.SetOutputs({y});
}

void LoadBroadcastStore_AfterAutofuse(ascir::ImplGraph& graph, bool is_f16) {
  auto x = graph.Find("x");
  x.attr.hint.compute_type = COMPUTE_DATA;
  x.attr.api.type = API_TYPE_BUFFER;
  x.attr.api.unit = UNIT_NONE;

  auto load = graph.Find("load");
  if (is_f16) {
    load.outputs[0].dtype = ge::DT_FLOAT16;
  } else {
    load.outputs[0].dtype = ge::DT_FLOAT;
  }

  load.attr.hint.compute_type = COMPUTE_LOAD;
  load.attr.api.type = API_TYPE_COMPUTE;
  load.attr.api.unit = UNIT_MTE;

  auto broadcast = graph.Find("broadcast");
  broadcast.attr.hint.compute_type = COMPUTE_BROADCAST;
  if (is_f16) {
    broadcast.outputs[0].dtype = ge::DT_FLOAT16;
  } else {
    broadcast.outputs[0].dtype = ge::DT_FLOAT;
  }

  broadcast.attr.api.type = API_TYPE_COMPUTE;
  broadcast.attr.api.unit = UNIT_VECTOR;

  auto store = graph.Find("store");
  store.attr.hint.compute_type = COMPUTE_STORE;
  if (is_f16) {
    store.outputs[0].dtype = ge::DT_FLOAT16;
  } else {
    store.outputs[0].dtype = ge::DT_FLOAT;
  }

  store.attr.api.type = API_TYPE_COMPUTE;
  store.attr.api.unit = UNIT_MTE;

  auto y = graph.Find("y");
  y.attr.hint.compute_type = COMPUTE_DATA;
  y.attr.api.type = API_TYPE_BUFFER;
  y.attr.api.unit = UNIT_NONE;

  // Scheduler
  auto z0 = load.attr.sched.axis[0];
  auto z1 = load.attr.sched.axis[1];

  auto [z0T, z0t] = graph.TileSplit(z0);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T.id);
  auto [z1T, z1t] = graph.TileSplit(z1);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }

    graph.ApplySplit(node, z0T.id, z0t.id);
    graph.ApplySplit(node, z0TB.id, z0Tb.id);
    graph.ApplySplit(node, z1T.id, z1t.id);
    graph.ApplyReorder(node, {z0TB.id, z0Tb.id, z1T.id, z0t.id, z1t.id});
  }

  // Vectorized/Loop axis
  load.attr.sched.loop_axis = z1T.id;
  load.outputs[0].vectorized_axis = {z0t.id, z1t.id};

  broadcast.attr.sched.loop_axis = z1T.id;
  broadcast.outputs[0].vectorized_axis = {z0t.id, z1t.id};

  store.attr.sched.loop_axis = z1T.id;
  store.outputs[0].vectorized_axis = {z0t.id, z1t.id};

  // Que/Buf alloc
  x.outputs[0].mem.tensor_id = 0;
  x.outputs[0].mem.alloc_type = ALLOC_TYPE_GLOBAL;
  x.outputs[0].mem.hardware = MEM_HARDWARE_GM;
  x.outputs[0].mem.position = POSITION_GM;
  x.outputs[0].buf.id = ID_NONE;
  x.outputs[0].que.id = ID_NONE;
  x.outputs[0].opt.ref_tensor = ID_NONE;
  x.outputs[0].opt.merge_scope = ID_NONE;

  load.outputs[0].mem.tensor_id = 1;
  load.outputs[0].mem.alloc_type = ALLOC_TYPE_QUEUE;
  load.outputs[0].mem.hardware = MEM_HARDWARE_UB;
  load.outputs[0].mem.position = POSITION_VECIN;
  load.outputs[0].buf.id = ID_NONE;
  load.outputs[0].que.id = 0;
  load.outputs[0].que.depth = 2;
  load.outputs[0].que.buf_num = 2;
  load.outputs[0].opt.ref_tensor = ID_NONE;
  load.outputs[0].opt.merge_scope = ID_NONE;

  broadcast.outputs[0].mem.tensor_id = 2;
  broadcast.outputs[0].mem.alloc_type = ALLOC_TYPE_QUEUE;
  broadcast.outputs[0].mem.hardware = MEM_HARDWARE_UB;
  broadcast.outputs[0].mem.position = POSITION_VECOUT;
  broadcast.outputs[0].buf.id = ID_NONE;
  broadcast.outputs[0].que.id = 1;
  broadcast.outputs[0].que.depth = 2;
  broadcast.outputs[0].que.buf_num = 2;
  broadcast.outputs[0].opt.ref_tensor = ID_NONE;
  broadcast.outputs[0].opt.merge_scope = ID_NONE;

  store.outputs[0].mem.tensor_id = 3;
  store.outputs[0].mem.alloc_type = ALLOC_TYPE_GLOBAL;
  store.outputs[0].mem.hardware = MEM_HARDWARE_GM;
  store.outputs[0].mem.position = POSITION_GM;
  store.outputs[0].buf.id = ID_NONE;
  store.outputs[0].que.id = ID_NONE;
  store.outputs[0].opt.ref_tensor = ID_NONE;
  store.outputs[0].opt.merge_scope = ID_NONE;
}
