#include "gtest/gtest.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "codegen_kernel.h"

TEST(CodegenKernel, Type_StrWillReturnTypeName) {
  codegen::Type t{"int"};

  EXPECT_EQ(t.Str(), "int");
}

TEST(CodegenKernel, Variable_StrWillReturnVarName) {
  codegen::Type t{"int"};
  codegen::Variable v{t, "x"};

  EXPECT_EQ(v.Str(), "x");
}

TEST(CodegenKernel, Variable_DefineWillReturnDefineWithInit) {
  codegen::Type t{"int"};
  codegen::Variable v{t, "x"};
  EXPECT_EQ(v.Define("100"), "int x = 100;");
}

TEST(CodegenKernel, Variable_AsArg) {
  codegen::Type t{"int"};
  codegen::Variable v{t, "x"};
  EXPECT_EQ(v.AsArg(), "int x");
}

TEST(CodegenKernel, Axis_StrWillReturnAxisName) {
  ascir::Axis axis{.name = "z0"};
  codegen::Axis codegen_axis(axis);
  EXPECT_EQ(codegen_axis.Str(), "z0");
}

TEST(CodegenKernel, Tiler_StrWillReturnTilingDataName) {
  codegen::Tiler tiler("tiling_data");
  EXPECT_EQ(codegen::Tiler(tiler).Str(), "tiling_data");
}

TEST(CodegenKernel, Tiler_SizeWillReturnTilingDataField) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  codegen::Tiler tiler("tiling_data");
  tiler.AddSizeVar(s0);

  EXPECT_EQ(tiler.Size(s0), "tiling_data.s0");
}

TEST(CodegenKernel, Tiler_SizeWhenHasTwoMoreNums_WillAddRoundBrackets) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  codegen::Tiler tiler("tiling_data");
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);

  EXPECT_EQ(tiler.Size(s0 * s1), "(tiling_data.s0 * tiling_data.s1)");
}

TEST(CodegenKernel, Tiler_Size) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  codegen::Tiler tiler("tiling_data");
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);

  EXPECT_EQ(tiler.Size(ascir::SizeExpr::Zero()), "0");
  EXPECT_EQ(tiler.Size(ascir::SizeExpr::One()), "1");

  EXPECT_EQ(tiler.Size(s0), "tiling_data.s0");
  EXPECT_EQ(tiler.Size(s0 * s1), "(tiling_data.s0 * tiling_data.s1)");
  EXPECT_EQ(tiler.Size(ascir::SizeExpr::One() / s0), "1 / tiling_data.s0");
  EXPECT_EQ(tiler.Size(ascir::SizeExpr::One() / (s0 * s1)), "1 / (tiling_data.s0 * tiling_data.s1)");
  EXPECT_EQ(tiler.Size(s0 / s1), "tiling_data.s0 / tiling_data.s1");
}

TEST(CodegenKernel, Tiler_TensorVectorizedSize) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s2{.id = 2, .name = "s2", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  ascir::Axis z0{.id = 0, .name = "z0", .type = ascir::Axis::AXIS_TYPE_TILE_OUTER, .size = s0};
  ascir::Axis z1{.id = 1, .name = "z1", .type = ascir::Axis::AXIS_TYPE_TILE_INNER, .size = s1};
  ascir::Axis z2{.id = 2, .name = "z2", .type = ascir::Axis::AXIS_TYPE_ORIGINAL, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  tensor.axis = {z0.id, z1.id, z2.id};
  tensor.vectorized_axis = {z1.id, z2.id};
  tensor.repeats = {z0.size, z1.size, z2.size};
  tensor.strides = {z1.size*z2.size, z2.size, ascir::SizeExpr::One()};

  EXPECT_EQ(tiler.TensorVectorizedSize(codegen::Tensor(tensor)), std::string{
    "(t.s1 - 1) * t.s2 + (t.s2 - 1) + 1"});
}

TEST(CodegenKernel, Tiler_TensorVectorizedSize_WhenNotVectorized) {
  codegen::Tiler tiler;
  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  EXPECT_EQ(tiler.TensorVectorizedSize(codegen::Tensor(tensor)), std::string{
    "1"});
}

TEST(CodegenKernel, TilingDataDefine) {
  codegen::GM_ADDR tiling("tiling");
  codegen::Tiler tiler("tiling_data");
  EXPECT_EQ(tiler.TilingDataDefine(tiling), "GET_TILING_DATA(tiling_data, tiling);\n");
}

TEST(CodegenKernel, Tiler_BlockOutterAxisDefine) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s2{.id = 2, .name = "s2", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  ascir::Axis z0{.id = 0, .name = "z0", .type = ascir::Axis::AXIS_TYPE_BLOCK_OUTER, .size = s0};
  ascir::Axis z1{.id = 1, .name = "z1", .type = ascir::Axis::AXIS_TYPE_BLOCK_OUTER, .size = s1};
  ascir::Axis z2{.id = 2, .name = "z2", .type = ascir::Axis::AXIS_TYPE_BLOCK_OUTER, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  auto result_code = tiler.BlockOutterAxisDefine();
  EXPECT_EQ(result_code, std::string{
     "int block_dim = GetBlockIdx();\n"
     "const int z0 = block_dim % (t.s0); block_dim /= t.s0;\n"
     "const int z1 = block_dim % (t.s1); block_dim /= t.s1;\n"
     "const int z2 = block_dim % (t.s2); block_dim /= t.s2;\n"
     });
}

TEST(CodegenKernel, Tiler_GetAxisVar) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::Axis z0{.id = 0, .name = "z0", .type = ascir::Axis::AXIS_TYPE_BLOCK_OUTER, .size = s0};

  codegen::Tiler tiler;
  tiler.AddSizeVar(s0);
  tiler.AddAxis(z0);

  EXPECT_EQ(tiler.GetAxis(z0.id).Str(), "z0");
}

TEST(CodegenKernel, Tensor_SetGlobalBuffer) {
  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);
  tensor.dtype = ge::DT_FLOAT16;

  codegen::Tensor t(tensor, "tensor");
  codegen::GM_ADDR gm("gm");

  EXPECT_EQ(t.SetGlobalBuffer(gm), "tensor.SetGlobalBuffer((__gm__ half*)gm);");
}

TEST(CodegenKernel, TPipe_InitTQueBuffers) {
  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  tensor.mem.position = ascir::POSITION_VECIN;
  tensor.mem.tensor_id = 0;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_QUEUE;
  tensor.que.id = 1;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(tensor);

  auto que = tpipe.ques.find(tensor.que.id);
  ASSERT_NE(que, tpipe.ques.end());
  EXPECT_EQ(tpipe.InitTQueBuffers(que->second), std::string {
    "tpipe.InitBuffer(q1, q1_buf_num, q1_size);"});
}

TEST(CodegenKernel, TPipe_InitTBufBuffer) {
  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  tensor.mem.position = ascir::POSITION_VECIN;
  tensor.mem.tensor_id = 0;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_BUFFER;
  tensor.buf.id = 1;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(tensor);

  auto buf = tpipe.bufs.find(tensor.buf.id);
  ASSERT_NE(buf, tpipe.bufs.end());
  EXPECT_EQ(tpipe.InitTBufBuffer(buf->second), std::string{
    "tpipe.InitBuffer(b1, b1_size);"});
}

TEST(CodegenKernel, TPipe_TensorSizeCalc_AllocFromBuf) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s2{.id = 2, .name = "s2", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  ascir::Axis z0{.id = 0, .name = "z0", .type = ascir::Axis::AXIS_TYPE_TILE_OUTER, .size = s0};
  ascir::Axis z1{.id = 1, .name = "z1", .type = ascir::Axis::AXIS_TYPE_TILE_INNER, .size = s1};
  ascir::Axis z2{.id = 2, .name = "z2", .type = ascir::Axis::AXIS_TYPE_ORIGINAL, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  tensor.axis = {z0.id, z1.id, z2.id};
  tensor.vectorized_axis = {z1.id, z2.id};
  tensor.repeats = {z0.size, z1.size, z2.size};
  tensor.strides = {z1.size*z2.size, z2.size, ascir::SizeExpr::One()};
  tensor.mem.tensor_id = 1;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_BUFFER;
  tensor.mem.position = ascir::POSITION_VECIN;
  tensor.opt.merge_scope = ascir::ID_NONE;
  tensor.buf.id = 2;

  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(tensor);

  EXPECT_EQ(tpipe.TensorSizeCalc(), std::string{
      "const uint32_t t1_size = (t.s1 - 1) * t.s2 + (t.s2 - 1) + 1;\n"
  });
}

TEST(CodegenKernel, TPipe_TensorSizeCalc_AllocFromQue) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s2{.id = 2, .name = "s2", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  ascir::Axis z0{.id = 0, .name = "z0", .type = ascir::Axis::AXIS_TYPE_TILE_OUTER, .size = s0};
  ascir::Axis z1{.id = 1, .name = "z1", .type = ascir::Axis::AXIS_TYPE_TILE_INNER, .size = s1};
  ascir::Axis z2{.id = 2, .name = "z2", .type = ascir::Axis::AXIS_TYPE_ORIGINAL, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  tensor.axis = {z0.id, z1.id, z2.id};
  tensor.vectorized_axis = {z1.id, z2.id};
  tensor.repeats = {z0.size, z1.size, z2.size};
  tensor.strides = {z1.size*z2.size, z2.size, ascir::SizeExpr::One()};
  tensor.mem.tensor_id = 1;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_QUEUE;
  tensor.mem.position = ascir::POSITION_VECIN;
  tensor.opt.merge_scope = ascir::ID_NONE;
  tensor.que.id = 2;
  tensor.que.depth = 3;
  tensor.que.buf_num = 4;

  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(tensor);

  EXPECT_EQ(tpipe.TensorSizeCalc(), std::string{
      "const uint32_t t1_size = (t.s1 - 1) * t.s2 + (t.s2 - 1) + 1;\n"
      "const uint32_t t1_que_depth = 3;\n"
      "const uint32_t t1_que_buf_num = 4;\n"
  });
}

TEST(CodegenKernel, TPipe_MergeScopeSizeCalc) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s2{.id = 2, .name = "s2", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  ascir::Axis z0{.id = 0, .name = "z0", .type = ascir::Axis::AXIS_TYPE_TILE_OUTER, .size = s0};
  ascir::Axis z1{.id = 1, .name = "z1", .type = ascir::Axis::AXIS_TYPE_TILE_INNER, .size = s1};
  ascir::Axis z2{.id = 2, .name = "z2", .type = ascir::Axis::AXIS_TYPE_ORIGINAL, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  tensor.axis = {z0.id, z1.id, z2.id};
  tensor.vectorized_axis = {z1.id, z2.id};
  tensor.repeats = {z0.size, z1.size, z2.size};
  tensor.strides = {z1.size*z2.size, z2.size, ascir::SizeExpr::One()};
  tensor.mem.position = ascir::POSITION_VECIN;

  tensor.opt.merge_scope = 1;

  codegen::TPipe tpipe("tpipe", tiler);

  tensor.dtype = ge::DT_FLOAT16;
  tensor.mem.tensor_id = 0;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_QUEUE;
  tensor.que.id = 2;
  tensor.que.depth = 3;
  tensor.que.buf_num = 4;
  tpipe.AddTensor(tensor);

  tensor.dtype = ge::DT_FLOAT;
  tensor.mem.tensor_id = 1;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_BUFFER;
  tensor.buf.id = 2;
  tpipe.AddTensor(tensor);

  EXPECT_EQ(tpipe.MergeScopeSizeCalc(), std::string{
      "const uint32_t m1_size = utils::Sum(t0_size * sizeof(half), t1_size * sizeof(float));\n"
      "const uint32_t m1_que_depth = utils::Max(t0_que_depth);\n"
      "const uint32_t m1_que_buf_num = utils::Max(t0_que_buf_num);\n"
  });
}

TEST(CodegenKernel, TPipe_LocalTBufAlloc) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s2{.id = 2, .name = "s2", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  ascir::Axis z0{.id = 0, .name = "z0", .type = ascir::Axis::AXIS_TYPE_TILE_OUTER, .size = s0};
  ascir::Axis z1{.id = 1, .name = "z1", .type = ascir::Axis::AXIS_TYPE_TILE_INNER, .size = s1};
  ascir::Axis z2{.id = 2, .name = "z2", .type = ascir::Axis::AXIS_TYPE_ORIGINAL, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  tensor.axis = {z0.id, z1.id, z2.id};
  tensor.vectorized_axis = {z1.id, z2.id};
  tensor.repeats = {z0.size, z1.size, z2.size};
  tensor.strides = {z1.size*z2.size, z2.size, ascir::SizeExpr::One()};
  tensor.mem.position = ascir::POSITION_VECIN;

  codegen::TPipe tpipe("tpipe", tiler);

  tensor.dtype = ge::DT_FLOAT16;
  tensor.mem.tensor_id = 0;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_BUFFER;
  tensor.buf.id = 1;
  tensor.opt.merge_scope = 1;
  tpipe.AddTensor(tensor);

  tensor.dtype = ge::DT_FLOAT;
  tensor.mem.tensor_id = 1;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_BUFFER;
  tensor.buf.id = 1;
  tensor.opt.merge_scope = ascir::ID_NONE;
  tpipe.AddTensor(tensor);

  EXPECT_EQ(tpipe.LocalTBufAlloc(), std::string{
    "const uint32_t b1_size = utils::Max(m1_size, t1_size * sizeof(float));\n"
    "TBuf<TPosition::VECIN> b1;\n"
    "tpipe.InitBuffer(b1, b1_size);\n"
    "\n"
  });
}

TEST(CodegenKernel, TPipe_LocalTQueAlloc) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s2{.id = 2, .name = "s2", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  ascir::Axis z0{.id = 0, .name = "z0", .type = ascir::Axis::AXIS_TYPE_TILE_OUTER, .size = s0};
  ascir::Axis z1{.id = 1, .name = "z1", .type = ascir::Axis::AXIS_TYPE_TILE_INNER, .size = s1};
  ascir::Axis z2{.id = 2, .name = "z2", .type = ascir::Axis::AXIS_TYPE_ORIGINAL, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  tensor.axis = {z0.id, z1.id, z2.id};
  tensor.vectorized_axis = {z1.id, z2.id};
  tensor.repeats = {z0.size, z1.size, z2.size};
  tensor.strides = {z1.size*z2.size, z2.size, ascir::SizeExpr::One()};
  tensor.mem.position = ascir::POSITION_VECIN;

  codegen::TPipe tpipe("tpipe", tiler);

  tensor.dtype = ge::DT_FLOAT16;
  tensor.mem.tensor_id = 0;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_QUEUE;
  tensor.que.id = 1;
  tensor.opt.merge_scope = 1;
  tpipe.AddTensor(tensor);

  tensor.dtype = ge::DT_FLOAT;
  tensor.mem.tensor_id = 1;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_QUEUE;
  tensor.que.id = 1;
  tensor.opt.merge_scope = ascir::ID_NONE;
  tpipe.AddTensor(tensor);

  EXPECT_EQ(tpipe.LocalTQueAlloc(), std::string{
    "const uint32_t q1_size = utils::Max(m1_size, t1_size * sizeof(float));\n"
    "const uint32_t q1_depth = utils::Max(m1_que_depth, t1_que_depth);\n"
    "const uint32_t q1_buf_num = utils::Max(m1_que_buf_num, t1_que_buf_num);\n"
    "TQue<TPosition::VECIN, q1_depth> q1;\n"
    "tpipe.InitBuffer(q1, q1_buf_num, q1_size);\n"
    "\n"
  });
}

TEST(CodegenKernel, Kernel_GlobalTensorInit) {
  ascir::ops::Data x_op("x");
  ascir::ops::Output y_op("y");
  ascir::ops::Load load_op("load");
  ascir::ops::Store store_op("store");

  x_op.y.dtype = ge::DT_FLOAT16;
  load_op.x = x_op;
  load_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = load_op;
  store_op.y.dtype = ge::DT_FLOAT16;
  y_op.x = store_op;

  ascir::ImplGraph graph("test_graph");
  graph.SetInputs({x_op});
  graph.SetOutputs({y_op});

  auto x = graph.Find("x");
  auto load = graph.Find("load");
  auto store = graph.Find("store");

  x.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_GLOBAL;
  x.outputs[0].mem.tensor_id = 0;
  load.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_BUFFER;
  load.outputs[0].mem.tensor_id = 1;
  store.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_GLOBAL;
  store.outputs[0].mem.tensor_id = 2;

  auto kernel = codegen::Kernel::ParseGraph(graph);

  EXPECT_EQ(kernel.GlobalTensorInit(), std::string{
    "GlobalTensor<half> x_y;\n"
    "x_y.SetGlobalBuffer((__gm__ half*)x);\n"
    "GlobalTensor<half> store_y;\n"
    "store_y.SetGlobalBuffer((__gm__ half*)y);\n"
  });
}

TEST(CodegenKernel, Kernel_KernelFunctionDeclare) {
  ascir::ops::Data x1("x1"), x2("x2"), x3("x3");
  ascir::ops::Output y1("y1"), y2("y2"), y3("y3");
  y1.x = x1;
  y2.x = x2;
  y3.x = x3;

  ascir::ImplGraph graph("test_kernel");
  graph.SetInputs({x1, x2, x3});
  graph.SetOutputs({y1, y2, y3});

  auto x1_node = graph.Find("x1");
  auto x2_node = graph.Find("x2");
  auto x3_node = graph.Find("x3");

  x1_node.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_GLOBAL;
  x1_node.outputs[0].mem.tensor_id = 0;
  x2_node.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_GLOBAL;
  x2_node.outputs[0].mem.tensor_id = 1;
  x3_node.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_GLOBAL;
  x3_node.outputs[0].mem.tensor_id = 2;

  auto kernel = codegen::Kernel::ParseGraph(graph);
  EXPECT_EQ(kernel.KernelFunctionDeclare(), std::string{
    "extern \"C\" __global__ __aicore__ void test_kernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3, GM_ADDR workspace, GM_ADDR tiling)"
  });
}

TEST(CodegenKernel, Kernel_LocalTensorQueBufAlloc) {
  ascir::ImplGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", ascir::SizeExpr({s0.id}));

  ascir::ops::Data x_op("x");
  ascir::ops::Load load_op("load");
  ascir::ops::Store store_op("store");
  ascir::ops::Output y_op("y");

  load_op.x = x_op;
  load_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = load_op;
  y_op.x = store_op;

  graph.SetInputs({x_op});
  graph.SetOutputs({y_op});

  auto x = graph.Find("x");
  auto load = graph.Find("load");
  auto store = graph.Find("store");
  auto y = graph.Find("y");

  x.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_GLOBAL;
  x.outputs[0].mem.tensor_id = 0;

  load.outputs[0].axis = {z0.id};
  load.outputs[0].vectorized_axis = {z0.id};
  load.outputs[0].repeats = {z0.size};
  load.outputs[0].strides = {ascir::SizeExpr::One()};
  load.outputs[0].mem.position = ascir::POSITION_VECIN;
  load.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_QUEUE;
  load.outputs[0].mem.tensor_id = 1;
  load.outputs[0].que.id = 0;
  load.outputs[0].que.depth = 2;
  load.outputs[0].que.buf_num = 2;
  load.outputs[0].opt.merge_scope = ascir::ID_NONE;

  store.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_GLOBAL;
  store.outputs[0].mem.tensor_id = 2;

  auto kernel = codegen::Kernel::ParseGraph(graph);

  EXPECT_EQ(kernel.LocalTensorQueBufAlloc(), std::string{
    "TPipe tpipe;\n"
    "\n"
    "const uint32_t load_y_size = (t.s0 - 1) + 1;\n"
    "const uint32_t load_y_que_depth = 2;\n"
    "const uint32_t load_y_que_buf_num = 2;\n"
    "\n"
    "\n"
    "const uint32_t q0_size = utils::Max(load_y_size * sizeof(half));\n"
    "const uint32_t q0_depth = utils::Max(load_y_que_depth);\n"
    "const uint32_t q0_buf_num = utils::Max(load_y_que_buf_num);\n"
    "TQue<TPosition::VECIN, q0_depth> q0;\n"
    "tpipe.InitBuffer(q0, q0_buf_num, q0_size);\n"
    "\n"
    "\n"
  });
}
