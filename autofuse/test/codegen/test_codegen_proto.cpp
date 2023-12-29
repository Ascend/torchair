#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "ascir_ops.h"
#include "codegen_proto.h"

using namespace ascir::ops;
using namespace codegen;

TEST(CodegenProto, ConstructFromGraph) {
  Data x("x");
  x.y.dtype = ge::DT_FLOAT16;

  Output y("y");
  y.x = x;
  y.y.dtype = ge::DT_FLOAT16;

  ascir::HintGraph graph("TestKernel");
  graph.SetInputs({x});
  graph.SetOutputs({y});

  auto proto = OpProto::FromGraph(graph);
  EXPECT_EQ(proto.op, "TestKernel");
  EXPECT_EQ(proto.language, "cpp");

  ASSERT_EQ(proto.input_desc.size(), 1);
  auto input_desc = proto.input_desc[0];
  EXPECT_EQ(input_desc.name, "x");
  EXPECT_EQ(input_desc.param_type, "required");
  EXPECT_EQ(input_desc.type[0], ge::DT_FLOAT16);
  EXPECT_EQ(input_desc.format[0], ge::FORMAT_ND);

  ASSERT_EQ(proto.output_desc.size(), 1);
  auto output_desc = proto.output_desc[0];
  EXPECT_EQ(output_desc.name, "y");
  EXPECT_EQ(output_desc.param_type, "required");
  EXPECT_EQ(output_desc.type[0], ge::DT_FLOAT16);
  EXPECT_EQ(output_desc.format[0], ge::FORMAT_ND);
}

TEST(CodegenProto, GenerateProto) {
    OpProto proto;
    proto.op = "TestKernel";
    proto.language = "cpp";
    proto.input_desc.push_back(OpParamDesc("x", "required", {ge::DT_FLOAT16}, {ge::FORMAT_ND}));
    proto.output_desc.push_back(OpParamDesc("y", "required", {ge::DT_FLOAT16}, {ge::FORMAT_ND}));

    auto proto_str = nlohmann::json{proto}.dump(4);
    auto result = nlohmann::json::parse(proto_str);

    auto op = result[0];
    EXPECT_EQ(op["op"], "TestKernel");
    EXPECT_EQ(op["language"], "cpp");

    auto input = op["input_desc"][0];
    EXPECT_EQ(input["name"], "x");
    EXPECT_EQ(input["param_type"], "required");
    EXPECT_EQ(input["type"][0], "fp16");
    EXPECT_EQ(input["format"][0], "ND");

    auto output = op["output_desc"][0];
    EXPECT_EQ(output["name"], "y");
    EXPECT_EQ(output["param_type"], "required");
    EXPECT_EQ(output["type"][0], "fp16");
    EXPECT_EQ(output["format"][0], "ND");
}
