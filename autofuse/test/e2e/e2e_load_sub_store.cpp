#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

using namespace std;
using namespace ascir;

void LoadSubStore_BeforeAutofuse(ascir::HintGraph &graph) {
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);

    ops::Data x1("x1");
    x1.attr.sched.exec_order = 0;
    x1.attr.sched.axis = {z0.id, z1.id, z2.id};
    x1.y.dtype = ge::DT_FLOAT16;
    x1.y.axis = {z0.id, z1.id, z2.id};
    x1.y.repeats = {s0, s1, s2};
    x1.y.strides = {s1*s2, s2, {}};

    ops::Data x2("x2");
    x2.attr.sched.exec_order = 1;
    x2.attr.sched.axis = {z0.id, z1.id, z2.id};
    x2.y.dtype = ge::DT_FLOAT16;
    x2.y.axis = {z0.id, z1.id, z2.id};
    x2.y.repeats = {s0, s1, s2};
    x2.y.strides = {s1*s2, s2, {}};

    ops::Load load1("load1");
    load1.x = x1;
    load1.attr.sched.exec_order = 2;
    load1.attr.sched.axis = {z0.id, z1.id, z2.id};
    load1.y.dtype = ge::DT_FLOAT16;
    load1.y.axis = {z0.id, z1.id, z2.id};
    load1.y.repeats = {s0, s1, s2};
    load1.y.strides = {s1*s2, s2, {}};

    ops::Load load2("load2");
    load2.x = x2;
    load2.attr.sched.exec_order = 3;
    load2.attr.sched.axis = {z0.id, z1.id, z2.id};
    load2.y.dtype = ge::DT_FLOAT16;
    load2.y.axis = {z0.id, z1.id, z2.id};
    load2.y.repeats = {s0, s1, s2};
    load2.y.strides = {s1*s2, s2, {}};

    ops::Sub sub("sub");
    sub.x1 = load1.y;
    sub.x2 = load2.y;
    sub.attr.sched.exec_order = 4;
    sub.attr.sched.axis = {z0.id, z1.id, z2.id};
    sub.y.dtype = ge::DT_FLOAT16;
    sub.y.axis = {z0.id, z1.id, z2.id};
    sub.y.repeats = {s0, s1, s2};
    sub.y.strides = {s1*s2, s2, {}};

    ops::Store store("store");
    store.x = sub.y;
    store.attr.sched.exec_order = 5;
    store.attr.sched.axis = {z0.id, z1.id, z2.id};
    store.y.dtype = ge::DT_FLOAT16;
    store.y.axis = {z0.id, z1.id, z2.id};
    store.y.repeats = {s0, s1, s2};
    store.y.strides = {s1*s2, s2, {}};

    ops::Output y("y");
    y.x = store.y;
    y.attr.sched.exec_order = 6;
    y.attr.sched.axis = {z0.id, z1.id, z2.id};
    y.y.dtype = ge::DT_FLOAT16;
    y.y.axis = {z0.id, z1.id, z2.id};
    y.y.repeats = {s0, s1, s2};
    y.y.strides = {s1*s2, s2, {}};

    graph.SetInputs({x1, x2});
    graph.SetOutputs({y});
}

void LoadSubStore_AfterInferOutput(ascir::HintGraph &graph) {
    auto x1 = graph.Find("x1");
    x1.attr.hint.compute_type = COMPUTE_DATA;

    auto x2 = graph.Find("x2");
    x2.attr.hint.compute_type = COMPUTE_DATA;

    auto load1 = graph.Find("load1");
    load1.outputs[0].dtype = ge::DT_FLOAT16;
    load1.attr.hint.compute_type = COMPUTE_LOAD;

    auto load2 = graph.Find("load2");
    load2.outputs[0].dtype = ge::DT_FLOAT16;
    load2.attr.hint.compute_type = COMPUTE_LOAD;

    auto sub = graph.Find("sub");
    sub.outputs[0].dtype =(ge::DataType)load1.outputs[0].dtype;
    sub.outputs[0].axis = load1.outputs[0].axis();
    sub.outputs[0].repeats = load1.outputs[0].repeats();
    sub.outputs[0].strides = load1.outputs[0].strides();
    sub.attr.hint.compute_type = COMPUTE_ELEWISE;

    auto store = graph.Find("store");
    store.outputs[0].dtype = (ge::DataType)sub.outputs[0].dtype;
    store.attr.hint.compute_type = COMPUTE_STORE;

    auto y = graph.Find("y");
    y.attr.hint.compute_type = COMPUTE_DATA;
}

void LoadSubStore_AfterGetApiInfo(ascir::ImplGraph &graph) {
    auto x1 = graph.Find("x1");
    x1.attr.api.type = API_TYPE_BUFFER;
    x1.attr.api.unit = UNIT_NONE;

    auto x2 = graph.Find("x2");
    x2.attr.api.type = API_TYPE_BUFFER;
    x2.attr.api.unit = UNIT_NONE;

    auto load1 = graph.Find("load1");
    load1.attr.api.type = API_TYPE_COMPUTE;
    load1.attr.api.unit = UNIT_MTE;

    auto load2 = graph.Find("load2");
    load2.attr.api.type = API_TYPE_COMPUTE;
    load2.attr.api.unit = UNIT_MTE;

    auto sub = graph.Find("sub");
    sub.attr.api.type = API_TYPE_COMPUTE;
    sub.attr.api.unit = UNIT_VECTOR;

    auto store = graph.Find("store");
    store.attr.api.type = API_TYPE_COMPUTE;
    store.attr.api.unit = UNIT_MTE;

    auto y = graph.Find("y");
    y.attr.api.type = API_TYPE_BUFFER;
    y.attr.api.unit = UNIT_NONE;
}

void LoadSubStore_AfterScheduler(ascir::ImplGraph &graph) {
    auto z0 = graph.axis[0].id;
    auto z1 = graph.axis[1].id;
    auto z2 = graph.axis[2].id;

    auto [z0B, z0b] = graph.BlockSplit(z0);
    auto [z1T, z1t] = graph.TileSplit(z1);
    vector<AxisId> vectorized_axis{z1t.id, z2};

    // ApplySplit on x, load, abs, store
    auto x1 = graph.Find("x1");
    graph.ApplySplit(x1, z0B.id, z0b.id, z0);
    graph.ApplySplit(x1, z1T.id, z1t.id, z1);
    x1.attr.sched.loop_axis = z1T.id;
    x1.outputs[0].vectorized_axis = vectorized_axis;

    auto x2 = graph.Find("x2");
    graph.ApplySplit(x2, z0B.id, z0b.id, z0);
    graph.ApplySplit(x2, z1T.id, z1t.id, z1);
    x2.attr.sched.loop_axis = z1T.id;
    x2.outputs[0].vectorized_axis = vectorized_axis;

    auto load1 = graph.Find("load1");
    graph.ApplySplit(load1, z0B.id, z0b.id, z0);
    graph.ApplySplit(load1, z1T.id, z1t.id, z1);
    load1.attr.sched.loop_axis = z1T.id;
    load1.outputs[0].vectorized_axis = vectorized_axis;

    auto load2 = graph.Find("load2");
    graph.ApplySplit(load2, z0B.id, z0b.id, z0);
    graph.ApplySplit(load2, z1T.id, z1t.id, z1);
    load2.attr.sched.loop_axis = z1T.id;
    load2.outputs[0].vectorized_axis = vectorized_axis;

    auto sub = graph.Find("sub");
    graph.ApplySplit(sub, z0B.id, z0b.id, z0);
    graph.ApplySplit(sub, z1T.id, z1t.id, z1);
    sub.attr.sched.loop_axis = z1T.id;
    sub.outputs[0].vectorized_axis = vectorized_axis;

    auto store = graph.Find("store");
    graph.ApplySplit(store, z0B.id, z0b.id, z0);
    graph.ApplySplit(store, z1T.id, z1t.id, z1);
    store.attr.sched.loop_axis = z1T.id;
    store.outputs[0].vectorized_axis = vectorized_axis;
}

void LoadSubStore_AfterQueBufAlloc(ascir::ImplGraph &graph) {
    auto x1 = graph.Find("x1");
    x1.outputs[0].mem.tensor_id = 0;
    x1.outputs[0].mem.alloc_type = ALLOC_TYPE_GLOBAL;
    x1.outputs[0].mem.hardware = MEM_HARDWARE_GM;
    x1.outputs[0].mem.position = POSITION_GM;
    x1.outputs[0].buf.id = ID_NONE;
    x1.outputs[0].que.id = ID_NONE;
    x1.outputs[0].opt.ref_tensor = ID_NONE;
    x1.outputs[0].opt.merge_scope = ID_NONE;

    auto x2 = graph.Find("x2");
    x2.outputs[0].mem.tensor_id = 1;
    x2.outputs[0].mem.alloc_type = ALLOC_TYPE_GLOBAL;
    x2.outputs[0].mem.hardware = MEM_HARDWARE_GM;
    x2.outputs[0].mem.position = POSITION_GM;
    x2.outputs[0].buf.id = ID_NONE;
    x2.outputs[0].que.id = ID_NONE;
    x2.outputs[0].opt.ref_tensor = ID_NONE;
    x2.outputs[0].opt.merge_scope = ID_NONE;

    auto load1 = graph.Find("load1");
    load1.outputs[0].mem.tensor_id = 2;
    load1.outputs[0].mem.alloc_type = ALLOC_TYPE_QUEUE;
    load1.outputs[0].mem.hardware = MEM_HARDWARE_UB;
    load1.outputs[0].mem.position = POSITION_VECIN;
    load1.outputs[0].buf.id = ID_NONE;
    load1.outputs[0].que.id = 0;
    load1.outputs[0].que.depth = 2;
    load1.outputs[0].que.buf_num = 2;
    load1.outputs[0].opt.ref_tensor = ID_NONE;
    load1.outputs[0].opt.merge_scope = ID_NONE;

    auto load2 = graph.Find("load2");
    load2.outputs[0].mem.tensor_id = 3;
    load2.outputs[0].mem.alloc_type = ALLOC_TYPE_QUEUE;
    load2.outputs[0].mem.hardware = MEM_HARDWARE_UB;
    load2.outputs[0].mem.position = POSITION_VECIN;
    load2.outputs[0].buf.id = ID_NONE;
    load2.outputs[0].que.id = 1;
    load2.outputs[0].que.depth = 2;
    load2.outputs[0].que.buf_num = 2;
    load2.outputs[0].opt.ref_tensor = ID_NONE;
    load2.outputs[0].opt.merge_scope = ID_NONE;

    auto sub = graph.Find("sub");
    sub.outputs[0].mem.tensor_id = 4;
    sub.outputs[0].mem.alloc_type = ALLOC_TYPE_QUEUE;
    sub.outputs[0].mem.hardware = MEM_HARDWARE_UB;
    sub.outputs[0].mem.position = POSITION_VECOUT;
    sub.outputs[0].buf.id = ID_NONE;
    sub.outputs[0].que.id = 2;
    sub.outputs[0].que.depth = 2;
    sub.outputs[0].que.buf_num = 2;
    sub.outputs[0].opt.ref_tensor = ID_NONE;
    sub.outputs[0].opt.merge_scope = ID_NONE;

    auto store = graph.Find("store");
    store.outputs[0].mem.tensor_id = 5;
    store.outputs[0].mem.alloc_type = ALLOC_TYPE_GLOBAL;
    store.outputs[0].mem.hardware = MEM_HARDWARE_GM;
    store.outputs[0].mem.position = POSITION_GM;
    store.outputs[0].buf.id = ID_NONE;
    store.outputs[0].que.id = ID_NONE;
    store.outputs[0].opt.ref_tensor = ID_NONE;
    store.outputs[0].opt.merge_scope = ID_NONE;
}