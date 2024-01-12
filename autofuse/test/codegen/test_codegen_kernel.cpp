#include "gtest/gtest.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "codegen_kernel.h"

using namespace ascir::ops;

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

TEST(CodegenKernel, Tiler_TensorOffset_WhenGlobalTensor_WillOffsetAll) {
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

  std::vector<ascir::AxisId> tensor_axis = {z0.id, z1.id, z2.id};
  std::vector<ascir::SizeExpr> stride = {s1 * s2, s2, ascir::SizeExpr::One()};

  EXPECT_EQ(tiler.Offset({}, tensor_axis, stride), std::string{"0"});
  EXPECT_EQ(tiler.Offset({z0.id}, tensor_axis, stride), std::string{"z0 * (t.s1 * t.s2)"});
  EXPECT_EQ(tiler.Offset({z0.id, z1.id}, tensor_axis, stride), std::string{"z0 * (t.s1 * t.s2) + z1 * t.s2"});
  EXPECT_EQ(tiler.Offset({z0.id, z1.id, z2.id}, tensor_axis, stride), std::string{"z0 * (t.s1 * t.s2) + z1 * t.s2 + z2"});
}

TEST(CodegenKernel, Tiler_TensorOffset_WhenLocalTensor_VectorizedOnCurrentAxis) {
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
  tensor.strides = {s1 * s2, s2, ascir::SizeExpr::One()};
  tensor.vectorized_axis = {z1.id, z2.id};

  codegen::Tensor t(tensor, "t");

  EXPECT_EQ(tiler.TensorVectorizedOffset({z0.id}, codegen::Tensor(tensor)), "0");
  EXPECT_EQ(tiler.TensorVectorizedOffset({z0.id, z1.id}, codegen::Tensor(tensor)), "z1 * t.s2");
  EXPECT_EQ(tiler.TensorVectorizedOffset({z0.id, z1.id, z2.id}, codegen::Tensor(tensor)), "z1 * t.s2 + z2");
}

TEST(CodegenKernel, Tiler_TensorOffset_WhenLocalTensor_VectorizedNestCurrentAxis) {
    GTEST_SKIP();
}

TEST(CodegenKernel, Tiler_TensorAlloc_WhenTensorFromQue_AndMerge) {
    GTEST_SKIP();
}

TEST(CodegenKernel, Tensor_SetGlobalBuffer) {
  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);
  tensor.dtype = ge::DT_FLOAT16;

  codegen::Tensor t(tensor, "tensor");
  codegen::GM_ADDR gm("gm");

  EXPECT_EQ(t.SetGlobalBuffer(gm), "tensor.SetGlobalBuffer((__gm__ half*)gm);");
}

TEST(CodegenKernel, TQue_AllocBuf) {
  codegen::TQue que(0, ascir::POSITION_VECIN);
  EXPECT_EQ(que.AllocBuf(), "LocalTensor<uint8_t> q0_buf = q0.AllocTensor<uint8_t>();");
}

TEST(CodegenKernel, TQue_EnqueBuf) {
  codegen::TQue que(0, ascir::POSITION_VECIN);
  EXPECT_EQ(que.EnqueBuf(), "q0.EnQue(q0_buf);");
}

TEST(CodegenKernel, TQue_DequeBuf) {
  codegen::TQue que(0, ascir::POSITION_VECIN);
  EXPECT_EQ(que.DequeBuf(), "LocalTensor<uint8_t> q0_buf = q0.DeQue<uint8_t>();");
}

TEST(CodegenKernel, TQue_FreeBuf) {
  codegen::TQue que(0, ascir::POSITION_VECIN);
  EXPECT_EQ(que.FreeBuf(), "q0.FreeTensor(q0_buf);");
}

TEST(CodegenKernel, TPipe_TensorAlloc_WhenTensorFromQue_AndNotMerge) {
  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  tensor.dtype = ge::DT_FLOAT16;
  tensor.mem.position = ascir::POSITION_VECIN;
  tensor.mem.tensor_id = 0;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_QUEUE;
  tensor.que.id = 1;
  tensor.opt.merge_scope = ascir::ID_NONE;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(tensor, "test_t");

  EXPECT_EQ(tpipe.TensorAlloc(tpipe.GetTensor(tensor.mem.tensor_id)), std::string{
    "LocalTensor<half> test_t;\n"
    "test_t.SetAddrWithOffset(q1_buf, 0);\n"
    });
}

TEST(CodegenKernel, TPipe_TensorAlloc_WhenTensorFromBuf_AndNotMerge) {
  ge::GeTensorDesc desc;
  ascir::TensorAttr tensor(&desc);

  tensor.dtype = ge::DT_FLOAT16;
  tensor.mem.position = ascir::POSITION_VECIN;
  tensor.mem.tensor_id = 0;
  tensor.mem.alloc_type = ascir::ALLOC_TYPE_BUFFER;
  tensor.buf.id = 1;
  tensor.opt.merge_scope = ascir::ID_NONE;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(tensor, "test_t");

  EXPECT_EQ(tpipe.TensorAlloc(tpipe.GetTensor(tensor.mem.tensor_id)), std::string{
    "LocalTensor<half> test_t;\n"
    "test_t.SetAddrWithOffset(b1_buf, 0);\n"
    });
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
    "LocalTensor<uint8_t> b1_buf = b1.Get<uint8_t>();\n"
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

TEST(CodegenKernel, ApiCall_Generate) {
  GTEST_SKIP();
}

TEST(CodegenKernel, Stage_AddCall_WillCollectInputOutputQues) {
  GTEST_SKIP();
}

TEST(CodegenKernel, Stage_AddCall_WillAddCall) {
  GTEST_SKIP();
}

TEST(CodegenKernel, Stage_WhenHasWriteQue_WillAllocInStart_AndEnqueInEnd) {
  Data x_op("x");
  Load load_op("load");
  load_op.x = x_op;

  ascir::Graph graph("test_graph");
  graph.SetInputs({x_op});

  auto load = graph.Find("load");
  load.attr.api.unit = ascir::UNIT_MTE;
  load.outputs[0].dtype = ge::DT_FLOAT16;
  load.outputs[0].mem.position = ascir::POSITION_VECIN;
  load.outputs[0].mem.tensor_id = 0;
  load.outputs[0].mem.position = ascir::POSITION_VECIN;
  load.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_QUEUE;
  load.outputs[0].que.id = 1;
  load.outputs[0].opt.merge_scope = ascir::ID_NONE;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load.outputs[0]);

  codegen::Stage stage(ascir::UNIT_MTE);
  stage.AddCall(load);

  EXPECT_EQ(stage.Generate(tpipe, vector<ascir::AxisId>{}), std::string{
    "{\n"
    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
    "LocalTensor<half> t0;\n"
    "t0.SetAddrWithOffset(q1_buf, 0);\n"
    "DataCopy(t0[0], t0[0], t0_size);\n"
    "q1.EnQue(q1_buf);\n"
    "}\n"
  });
}

TEST(CodegenKernel, Stage_WhenHasReadQue_WillDequeInStart_AndFreeInEnd) {
  Data x_op("x");
  Load load_op("load");
  Abs abs_op("abs");

  load_op.x = x_op;
  abs_op.x = load_op;

  ascir::Graph graph("test_graph");
  graph.SetInputs({x_op});

  auto load = graph.Find("load");
  load.attr.api.unit = ascir::UNIT_MTE;
  load.outputs[0].dtype = ge::DT_FLOAT16;
  load.outputs[0].mem.position = ascir::POSITION_VECIN;
  load.outputs[0].mem.tensor_id = 0;
  load.outputs[0].mem.position = ascir::POSITION_VECIN;
  load.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_QUEUE;
  load.outputs[0].que.id = 1;
  load.outputs[0].opt.merge_scope = ascir::ID_NONE;

  auto abs = graph.Find("abs");
  abs.attr.api.unit = ascir::UNIT_VECTOR;
  abs.outputs[0].dtype = ge::DT_FLOAT16;
  abs.outputs[0].mem.position = ascir::POSITION_VECOUT;
  abs.outputs[0].mem.tensor_id = 1;
  abs.outputs[0].mem.alloc_type = ascir::ALLOC_TYPE_QUEUE;
  abs.outputs[0].que.id = 2;
  abs.outputs[0].opt.merge_scope = ascir::ID_NONE;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load.outputs[0]);
  tpipe.AddTensor(abs.outputs[0]);

  codegen::Stage stage(ascir::UNIT_VECTOR);
  stage.AddCall(abs);

  EXPECT_EQ(stage.Generate(tpipe, vector<ascir::AxisId>{}), std::string{
    "{\n"
    "LocalTensor<uint8_t> q1_buf = q1.DeQue<uint8_t>();\n"
    "LocalTensor<uint8_t> q2_buf = q2.AllocTensor<uint8_t>();\n"
    "LocalTensor<half> t0;\n"
    "t0.SetAddrWithOffset(q1_buf, 0);\n"
    "LocalTensor<half> t1;\n"
    "t1.SetAddrWithOffset(q2_buf, 0);\n"
    "Abs(t1[0], t0[0], t0_size);\n"
    "q2.EnQue(q2_buf);\n"
    "q1.FreeTensor(q1_buf);\n"
    "}\n"
  });
}

TEST(CodegenKernel, StageGenerate_WillNotDuplicatAllocTensorInSameStage) {
  GTEST_SKIP();
}

TEST(CodegenKernel, Looper_WillCreateNestedLoop_OnLoopAxis) {
  ascir::ImplGraph graph("test_graph");
  ascir::ops::Data x_op("x");
  graph.SetInputs({x_op});

  ascir::AxisId H = 0;
  ascir::AxisId W = 1;

  auto x = graph.Find("x");
  x.attr.sched.axis = {H, W};
  x.attr.sched.loop_axis = W;

  codegen::Looper looper;
  looper.InitRootLoop();
  looper.AddNode(x);
  looper.EndRootLoop();

  auto root = looper.loops[looper.root_loop];
  ASSERT_EQ(root.body.size(), 1);
  auto [result_loop_h_type, result_loop_h] = root.body[0];
  EXPECT_EQ(result_loop_h_type, codegen::Loop::LOOP);
  EXPECT_EQ(looper.loops[result_loop_h].axis, H);
  ASSERT_EQ(looper.loops[result_loop_h].body.size(), 1);

  auto [result_loop_w_type, result_loop_w] = looper.loops[result_loop_h].body[0];
  EXPECT_EQ(result_loop_w_type, codegen::Loop::LOOP);
  EXPECT_EQ(looper.loops[result_loop_w].axis, W);
  ASSERT_EQ(looper.loops[result_loop_w].body.size(), 1);

  auto [stage_type, stage] = looper.loops[result_loop_w].body[0];
  EXPECT_EQ(stage_type, codegen::Loop::STAGE);
  EXPECT_EQ(looper.stages[stage].unit, x.attr.api.unit);
}

TEST(CodegenKernel, Looper_WhenTwoNodeDifferenceLoopAxis_WillCreateTwoLoop) {
  ascir::ImplGraph graph("test_graph");
  ascir::ops::Data A_op("a");
  ascir::ops::Data B_op("b");
  graph.SetInputs({A_op, B_op});

  ascir::AxisId K = 0;
  ascir::AxisId M = 1;
  ascir::AxisId N = 2;

  auto a = graph.Find("a");
  a.attr.api.unit = ascir::UNIT_MTE;
  a.attr.sched.axis = {K, M};
  a.attr.sched.loop_axis = M;

  auto b = graph.Find("b");
  b.attr.api.unit = ascir::UNIT_VECTOR;
  b.attr.sched.axis = {K, N};
  b.attr.sched.loop_axis = N;

  codegen::Looper looper;
  looper.InitRootLoop();
  looper.AddNode(a);
  looper.AddNode(b);
  looper.EndRootLoop();

  auto root = looper.loops[looper.root_loop];
  ASSERT_EQ(root.body.size(), 1);

  auto [result_loop_k_type, result_loop_k] = root.body[0];
  EXPECT_EQ(result_loop_k_type, codegen::Loop::LOOP);
  EXPECT_EQ(looper.loops[result_loop_k].axis, K);
  ASSERT_EQ(looper.loops[result_loop_k].body.size(), 2);

  auto [result_loop_m_type, result_loop_m] = looper.loops[result_loop_k].body[0];
  EXPECT_EQ(result_loop_m_type, codegen::Loop::LOOP);
  EXPECT_EQ(looper.loops[result_loop_m].axis, M);
  ASSERT_EQ(looper.loops[result_loop_m].body.size(), 1);

  auto [stage_a_type, stage_a] = looper.loops[result_loop_m].body[0];
  EXPECT_EQ(stage_a_type, codegen::Loop::STAGE);
  EXPECT_EQ(looper.stages[stage_a].unit, a.attr.api.unit);

  auto [result_loop_n_type, result_loop_n] = looper.loops[result_loop_k].body[1];
  EXPECT_EQ(result_loop_n_type, codegen::Loop::LOOP);
  EXPECT_EQ(looper.loops[result_loop_n].axis, N);
  ASSERT_EQ(looper.loops[result_loop_n].body.size(), 1);

  auto [stage_b_type, stage_b] = looper.loops[result_loop_n].body[0];
  EXPECT_EQ(stage_b_type, codegen::Loop::STAGE);
  EXPECT_EQ(looper.stages[stage_b].unit, b.attr.api.unit);
}

TEST(CodegenKernel, Looper_WhenTwoNodeSameUnit_WillCreateOneStage) {
  ascir::ImplGraph graph("test_graph");
  ascir::ops::Data A_op("a");
  ascir::ops::Data B_op("b");
  graph.SetInputs({A_op, B_op});

  ascir::AxisId K = 0;

  auto a = graph.Find("a");
  a.attr.api.unit = ascir::UNIT_MTE;
  a.attr.sched.axis = {K};
  a.attr.sched.loop_axis = K;

  auto b = graph.Find("b");
  b.attr.api.unit = ascir::UNIT_MTE;
  b.attr.sched.axis = {K};
  b.attr.sched.loop_axis = K;

  codegen::Looper looper;
  looper.InitRootLoop();
  looper.AddNode(a);
  looper.AddNode(b);
  looper.EndRootLoop();

  auto root = looper.loops[looper.root_loop];
  ASSERT_EQ(root.body.size(), 1);

  auto [result_loop_k_type, result_loop_k] = root.body[0];
  EXPECT_EQ(result_loop_k_type, codegen::Loop::LOOP);
  EXPECT_EQ(looper.loops[result_loop_k].axis, K);
  ASSERT_EQ(looper.loops[result_loop_k].body.size(), 1);

  auto [stage_mte_type, stage_mte] = looper.loops[result_loop_k].body[0];
  EXPECT_EQ(stage_mte_type, codegen::Loop::STAGE);
  EXPECT_EQ(looper.stages[stage_mte].unit, a.attr.api.unit);

  auto stage = looper.stages[stage_mte];
  ASSERT_EQ(stage.calls.size(), 2);
}

TEST(CodegenKernel, Looper_WillCreateTwoStage_WhenTwoNodeDiffUnit) {
  ascir::ImplGraph graph("test_graph");
  ascir::ops::Data A_op("a");
  ascir::ops::Data B_op("b");
  graph.SetInputs({A_op, B_op});

  ascir::AxisId K = 0;

  auto a = graph.Find("a");
  a.attr.api.unit = ascir::UNIT_MTE;
  a.attr.sched.axis = {K};
  a.attr.sched.loop_axis = K;

  auto b = graph.Find("b");
  b.attr.api.unit = ascir::UNIT_VECTOR;
  b.attr.sched.axis = {K};
  b.attr.sched.loop_axis = K;

  codegen::Looper looper;
  looper.InitRootLoop();
  looper.AddNode(a);
  looper.AddNode(b);
  looper.EndRootLoop();

  auto root = looper.loops[looper.root_loop];
  ASSERT_EQ(root.body.size(), 1);

  auto [result_loop_k_type, result_loop_k] = root.body[0];
  EXPECT_EQ(result_loop_k_type, codegen::Loop::LOOP);
  EXPECT_EQ(looper.loops[result_loop_k].axis, K);
  ASSERT_EQ(looper.loops[result_loop_k].body.size(), 2);

  auto [stage_a_type, stage_a] = looper.loops[result_loop_k].body[0];
  EXPECT_EQ(stage_a_type, codegen::Loop::STAGE);
  EXPECT_EQ(looper.stages[stage_a].unit, a.attr.api.unit);
  EXPECT_EQ(looper.stages[stage_a].calls.size(), 1);

  auto [stage_b_type, stage_b] = looper.loops[result_loop_k].body[1];
  EXPECT_EQ(stage_b_type, codegen::Loop::STAGE);
  EXPECT_EQ(looper.stages[stage_b].unit, b.attr.api.unit);
  EXPECT_EQ(looper.stages[stage_b].calls.size(), 1);
}

TEST(CodegenKernel, Looper_GenerateLoop_WhenNestedLoop) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  ascir::Axis z0{.id = 0, .name = "z0", .size = s0};
  ascir::Axis z1{.id = 1, .name = "z1", .size = s1};

  codegen::Tiler tiler;
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);

  codegen::Looper looper;
  looper.InitRootLoop();
  looper.EnterLoop(z0.id);
  looper.EnterLoop(z1.id);
  looper.EndRootLoop();

  codegen::TPipe tpipe("t", tiler);
  EXPECT_EQ(looper.GenerateLoop(tiler, tpipe), std::string{
    "for (int z0 = 0; z0 < t.s0; z0++) {\n"
    "for (int z1 = 0; z1 < t.s1; z1++) {\n"
    "}\n"
    "}\n"
  });
}

TEST(CodegenKernel, Looper_GenerateLoop_WhenTwoLoop) {
  ascir::SizeVar s0{.id = 0, .name = "s0", .type = ascir::SizeVar::SIZE_TYPE_VAR};
  ascir::SizeVar s1{.id = 1, .name = "s1", .type = ascir::SizeVar::SIZE_TYPE_VAR};

  ascir::Axis z0{.id = 0, .name = "z0", .size = s0};
  ascir::Axis z1{.id = 1, .name = "z1", .size = s1};

  codegen::Tiler tiler;
  tiler.AddSizeVar(s0);
  tiler.AddSizeVar(s1);
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);

  codegen::Looper looper;
  looper.InitRootLoop();
  looper.EnterLoop(z0.id);
  looper.ExitLoop();
  looper.EnterLoop(z1.id);
  looper.EndRootLoop();

  codegen::TPipe tpipe("t", tiler);
  EXPECT_EQ(looper.GenerateLoop(tiler, tpipe), std::string{
    "for (int z0 = 0; z0 < t.s0; z0++) {\n"
    "}\n"
    "for (int z1 = 0; z1 < t.s1; z1++) {\n"
    "}\n"
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
