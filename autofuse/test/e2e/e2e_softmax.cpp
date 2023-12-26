#include "ascir.h"
#include "ascir_ops.h"

using namespace ascir;
using namespace ascir::ops;

void Softmax_BeforeAutofuse(ascir::HintGraph &graph) {
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
  arg4_1.y.dtype = ge::DT_FLOAT16;

  // buf0
  Data buf0("buf0");
  Load b0_load("b0_load");
  b0_load.x = arg4_1;
  b0_load.attr.sched.exec_order = exec_order++;
  b0_load.attr.sched.axis = axis;
  b0_load.y.dtype = ge::DT_FLOAT16;
  b0_load.y.axis = axis;
  b0_load.y.repeats = {s0*s1*s2, s3};
  b0_load.y.strides = {s3, ONE};

  Max b0_max("b0_max");
  b0_max.x = b0_load.y;
  b0_max.attr.sched.exec_order = exec_order++;
  b0_max.attr.sched.axis = axis;
  b0_max.y.dtype = ge::DT_FLOAT16;
  b0_max.y.axis = axis;
  b0_max.y.repeats = {s0*s1*s2, s3};
  b0_max.y.strides = {ONE, ZERO};

  Store b0_store("b0_store");
  b0_store.x = b0_max.y;
  b0_store.attr.sched.exec_order = exec_order++;
  b0_store.attr.sched.axis = axis;
  b0_store.y.dtype = ge::DT_FLOAT16;
  b0_store.y.axis = axis;
  b0_store.y.repeats = {s0*s1*s2, s3};
  b0_store.y.strides = {ONE, ZERO};

  buf0.x = b0_store.y;
  buf0.attr.sched.exec_order = exec_order++;
  buf0.y.dtype = ge::DT_FLOAT16;

  Data buf1("buf1");
  Load b1_load("b1_load");
  b1_load.x = arg4_1;
  b1_load.attr.sched.exec_order = exec_order++;
  b1_load.attr.sched.axis = axis;
  b1_load.y.dtype = ge::DT_FLOAT16;
  b1_load.y.axis = axis;
  b1_load.y.repeats = {s0*s1*s2, s3};
  b1_load.y.strides = {s3, ONE};

  Load b1_load1("b1_load1");
  b1_load1.x = buf0.y;
  b1_load1.attr.sched.exec_order = exec_order++;
  b1_load1.attr.sched.axis = axis;
  b1_load1.y.dtype = ge::DT_FLOAT16;
  b1_load1.y.axis = axis;
  b1_load1.y.repeats = {s0*s1*s2, s3};
  b1_load1.y.strides = {ONE, ZERO};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b1_load1.y;
  b1_broadcast.attr.sched.exec_order = exec_order++;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  b1_broadcast.y.axis = axis;
  b1_broadcast.y.repeats = {s0*s1*s2, s3};
  b1_broadcast.y.strides = {s3, ONE};

  Sub b1_sub("b1_sub");
  b1_sub.x1 = b1_load.y;
  b1_sub.x2 = b1_broadcast.y;
  b1_sub.attr.sched.exec_order = exec_order++;
  b1_sub.attr.sched.axis = axis;
  b1_sub.y.dtype = ge::DT_FLOAT16;
  b1_sub.y.axis = axis;
  b1_sub.y.repeats = {s0*s1*s2, s3};
  b1_sub.y.strides = {s3, ONE};

  Exp b1_exp("b1_exp");
  b1_exp.x = b1_sub.y;
  b1_exp.attr.sched.exec_order = exec_order++;
  b1_exp.attr.sched.axis = axis;
  b1_exp.y.dtype = ge::DT_FLOAT16;
  b1_exp.y.axis = axis;
  b1_exp.y.repeats = {s0*s1*s2, s3};
  b1_exp.y.strides = {s3, ONE};

  Store b1_store("b1_store");
  b1_store.x = b1_exp.y;
  b1_store.attr.sched.exec_order = exec_order++;
  b1_store.attr.sched.axis = axis;
  b1_store.y.dtype = ge::DT_FLOAT16;
  b1_store.y.axis = axis;
  b1_store.y.repeats = {s0*s1*s2, s3};
  b1_store.y.strides = {s3, ONE};

  buf1.x = b1_store.y;
  buf1.attr.sched.exec_order = exec_order++;
  buf1.y.dtype = ge::DT_FLOAT16;

  Data buf2("buf2");
  Load b2_load("b2_load");
  b2_load.x = buf1;
  b2_load.attr.sched.exec_order = exec_order++;
  b2_load.attr.sched.axis = axis;
  b2_load.y.dtype = ge::DT_FLOAT16;
  b2_load.y.axis = axis;
  b2_load.y.repeats = {s0*s1*s2, s3};
  b2_load.y.strides = {s3, ONE};

  Sum b2_sum("b2_sum");
  b2_sum.x = b2_load.y;
  b2_sum.attr.sched.exec_order = exec_order++;
  b2_sum.attr.sched.axis = axis;
  b2_sum.y.dtype = ge::DT_FLOAT16;
  b2_sum.y.axis = axis;
  b2_sum.y.repeats = {s0*s1*s2, s3};
  b2_sum.y.strides = {ONE, ZERO};

  Store b2_store("b2_store");
  b2_store.x = b2_sum.y;
  b2_store.attr.sched.exec_order = exec_order++;
  b2_store.attr.sched.axis = axis;
  b2_store.y.dtype = ge::DT_FLOAT16;
  b2_store.y.axis = axis;
  b2_store.y.repeats = {s0*s1*s2, s3};
  b2_store.y.strides = {ONE, ZERO};

  buf2.x = b2_store.y;
  buf2.attr.sched.exec_order = exec_order++;
  buf2.y.dtype = ge::DT_FLOAT16;

  Data buf3("buf3");
  Load b3_load("b3_load");
  b3_load.x = buf1;
  b3_load.attr.sched.exec_order = exec_order++;
  b3_load.attr.sched.axis = axis;
  b3_load.y.dtype = ge::DT_FLOAT16;
  b3_load.y.axis = axis;
  b3_load.y.repeats = {s0*s1*s2, s3};
  b3_load.y.strides = {s3, ONE};

  Load b3_load1("b3_load1");
  b3_load1.x = buf2;
  b3_load1.attr.sched.exec_order = exec_order++;
  b3_load1.attr.sched.axis = axis;
  b3_load1.y.dtype = ge::DT_FLOAT16;
  b3_load1.y.axis = axis;
  b3_load1.y.repeats = {s0*s1*s2, s3};
  b3_load1.y.strides = {ONE, ZERO};

  Broadcast b3_broadcast("b3_broadcast");
  b3_broadcast.x = b3_load1.y;
  b3_broadcast.attr.sched.exec_order = exec_order++;
  b3_broadcast.attr.sched.axis = axis;
  b3_broadcast.y.dtype = ge::DT_FLOAT16;
  b3_broadcast.y.axis = axis;
  b3_broadcast.y.repeats = {s0 * s1 * s2, s3};
  b3_broadcast.y.strides = {s3, ONE};

  Div b3_div("b3_div");
  b3_div.x1 = b3_load.y;
  b3_div.x2 = b3_broadcast.y;
  b3_div.attr.sched.exec_order = exec_order++;
  b3_div.attr.sched.axis = axis;
  b3_div.y.dtype = ge::DT_FLOAT16;
  b3_div.y.axis = axis;
  b3_div.y.repeats = {s0*s1*s2, s3};
  b3_div.y.strides = {s3, ONE};

  Store b3_store("b3_store");
  b3_store.x = b3_div.y;
  b3_store.attr.sched.exec_order = exec_order++;
  b3_store.attr.sched.axis = axis;
  b3_store.y.dtype = ge::DT_FLOAT16;
  b3_store.y.axis = axis;
  b3_store.y.repeats = {s0*s1*s2, s3};
  b3_store.y.strides = {s3, ONE};

  buf3.x = b3_store.y;
  buf3.attr.sched.exec_order = exec_order++;
  buf3.y.dtype = ge::DT_FLOAT16;

  graph.SetInputs({arg4_1});
  graph.SetOutputs({buf3});
}
