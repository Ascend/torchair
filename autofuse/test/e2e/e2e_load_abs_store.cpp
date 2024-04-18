#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

using namespace std;
using namespace ascir;

void LoadAbsStore_BeforeAutofuse(ascir::HintGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ops::Data x("x");
  x.attr.sched.exec_order = 0;
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.y.dtype = ge::DT_FLOAT16;
  x.y.axis = {z0.id, z1.id, z2.id};

  ops::Load load("load");
  load.x = x;
  load.attr.sched.exec_order = 1;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.y.axis = {z0.id, z1.id, z2.id};
  load.y.repeats = {s0, s1, s2};
  load.y.strides = {s1*s2, s2, {}};

  ops::Abs abs("abs");
  abs.x = load.y;
  abs.attr.sched.exec_order = 2;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};

  ops::Store store("store");
  store.x = abs.y;
  store.attr.sched.exec_order = 3;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  store.y.axis = {z0.id, z1.id, z2.id};
  store.y.repeats = {s0, s1, s2};
  store.y.strides = {s1*s2, s2, {}};

  ops::Output y("y");
  y.x = store.y;
  y.attr.sched.exec_order = 4;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  y.y.axis = {z0.id, z1.id, z2.id};

  graph.SetInputs({x});
  graph.SetOutputs({y});
}

void LoadAbsStore_AfterInferOutput(ascir::HintGraph &graph) {
  auto x = graph.Find("x");
  x.attr.hint.compute_type = COMPUTE_DATA;

  auto load = graph.Find("load");
  load.outputs[0].dtype = ge::DT_FLOAT16;
  load.attr.hint.compute_type = COMPUTE_LOAD;

  auto abs = graph.Find("abs");
  abs.outputs[0].dtype =(ge::DataType)load.outputs[0].dtype;
  abs.outputs[0].axis = load.outputs[0].axis();
  abs.outputs[0].repeats = load.outputs[0].repeats();
  abs.outputs[0].strides = load.outputs[0].strides();
  abs.attr.hint.compute_type = COMPUTE_ELEWISE;

  auto store = graph.Find("store");
  store.outputs[0].dtype = (ge::DataType)abs.outputs[0].dtype;
  store.attr.hint.compute_type = COMPUTE_STORE;

  auto y = graph.Find("y");
  y.attr.hint.compute_type = COMPUTE_DATA;
}

void LoadAbsStore_AfterGetApiInfo(ascir::ImplGraph &graph) {
  auto x = graph.Find("x");
  x.attr.api.type = API_TYPE_BUFFER;
  x.attr.api.unit = UNIT_NONE;

  auto load = graph.Find("load");
  load.attr.api.type = API_TYPE_COMPUTE;
  load.attr.api.unit = UNIT_MTE;

  auto abs = graph.Find("abs");
  abs.attr.api.type = API_TYPE_COMPUTE;
  abs.attr.api.unit = UNIT_VECTOR;

  auto store = graph.Find("store");
  store.attr.api.type = API_TYPE_COMPUTE;
  store.attr.api.unit = UNIT_MTE;

  auto y = graph.Find("y");
  y.attr.api.type = API_TYPE_BUFFER;
  y.attr.api.unit = UNIT_NONE;
}

void LoadAbsStore_AfterScheduler(ascir::ImplGraph &graph) {
  auto z0 = graph.axis[0].id;
  auto z1 = graph.axis[1].id;
  auto z2 = graph.axis[2].id;

  auto [z1T, z1t] = graph.TileSplit(z1);
  std::vector<AxisId> axes{z0, z1T.id};
  auto block_axis = graph.MergeAxis(axes);
  auto [z0B, z0b] = graph.BlockSplit(block_axis.id);
  vector<AxisId> vectorized_axis{z1t.id, z2};

  // ApplySplit on load, abs, store
  auto load = graph.Find("load");
  graph.ApplySplit(load, z1T.id, z1t.id, z1);
  graph.ApplyMerge(load, block_axis.id);
  graph.ApplySplit(load, z0B.id, z0b.id, block_axis.id);
  load.attr.sched.loop_axis = z0b.id;
  load.outputs[0].vectorized_axis = vectorized_axis;

  auto abs = graph.Find("abs");
  graph.ApplySplit(abs, z1T.id, z1t.id, z1);
  graph.ApplyMerge(abs, block_axis.id);
  graph.ApplySplit(abs, z0B.id, z0b.id, block_axis.id);
  abs.attr.sched.loop_axis = z0b.id;
  abs.outputs[0].vectorized_axis = vectorized_axis;

  auto store = graph.Find("store");
  graph.ApplySplit(store, z1T.id, z1t.id, z1);
  graph.ApplyMerge(store, block_axis.id);
  graph.ApplySplit(store, z0B.id, z0b.id, block_axis.id);
  store.attr.sched.loop_axis = z0b.id;
  store.outputs[0].vectorized_axis = vectorized_axis;
}

void LoadAbsStore_AfterQueBufAlloc(ascir::ImplGraph &graph) {
  auto x = graph.Find("x");
  x.outputs[0].mem.tensor_id = 0;
  x.outputs[0].mem.alloc_type = ALLOC_TYPE_GLOBAL;
  x.outputs[0].mem.hardware = MEM_HARDWARE_GM;
  x.outputs[0].mem.position = POSITION_GM;
  x.outputs[0].buf.id = ID_NONE;
  x.outputs[0].que.id = ID_NONE;
  x.outputs[0].opt.ref_tensor = ID_NONE;
  x.outputs[0].opt.merge_scope = ID_NONE;

  auto load = graph.Find("load");
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

  auto abs = graph.Find("abs");
  abs.outputs[0].mem.tensor_id = 2;
  abs.outputs[0].mem.alloc_type = ALLOC_TYPE_QUEUE;
  abs.outputs[0].mem.hardware = MEM_HARDWARE_UB;
  abs.outputs[0].mem.position = POSITION_VECOUT;
  abs.outputs[0].buf.id = ID_NONE;
  abs.outputs[0].que.id = 1;
  abs.outputs[0].que.depth = 2;
  abs.outputs[0].que.buf_num = 2;
  abs.outputs[0].opt.ref_tensor = ID_NONE;
  abs.outputs[0].opt.merge_scope = ID_NONE;

  auto store = graph.Find("store");
  store.outputs[0].mem.tensor_id = 3;
  store.outputs[0].mem.alloc_type = ALLOC_TYPE_GLOBAL;
  store.outputs[0].mem.hardware = MEM_HARDWARE_GM;
  store.outputs[0].mem.position = POSITION_GM;
  store.outputs[0].buf.id = ID_NONE;
  store.outputs[0].que.id = ID_NONE;
  store.outputs[0].opt.ref_tensor = ID_NONE;
  store.outputs[0].opt.merge_scope = ID_NONE;
}