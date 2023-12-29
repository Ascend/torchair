#include "codegen_proto.h"

#include "ascir.h"
#include "ascir_ops.h"

using namespace ascir;
using namespace codegen;

OpParamDesc::OpParamDesc(const std::string &name, const std::string &param_type, const std::vector<ge::DataType> &type,
                         const std::vector<ge::Format> &format)
    : name(name), param_type(param_type), type(type), format(format) {}

OpProto OpProto::FromGraph(const ascir::HintGraph &graph) {
  OpProto proto;
  proto.op  = graph.GetName();
  proto.language = "cpp";

  for (auto input : graph.GraphInputs()) {
    if (!ops::IsOps<ops::Data>(input)) {
      throw std::runtime_error("Graph input should be Data");
    }

    proto.input_desc.emplace_back(OpParamDesc(input->GetName(), "required", {input.outputs[0].dtype}, {ge::FORMAT_ND}));
  }

  for (auto output : graph.GraphOutputs()) {
    if (!ops::IsOps<ops::Output>(output)) {
      throw std::runtime_error("Graph output should be Output");
    }

    proto.output_desc.emplace_back(
        OpParamDesc(output->GetName(), "required", {output.outputs[0].dtype}, {ge::FORMAT_ND}));
  }
  return proto;
}

void nlohmann::to_json(nlohmann::json &j, const OpParamDesc &param) {
  j = json{{"name", param.name}, {"param_type", param.param_type}};

  const std::string dtypes[] = {[ge::DT_FLOAT] = "fp32", [ge::DT_FLOAT16] = "fp16"};
  for (auto dtype : param.type) {
    if (dtype > sizeof(dtypes) / sizeof(std::string)) {
        throw std::runtime_error("Unsupported dtype");
    }

    j["type"].push_back(dtypes[dtype]);
  }

  const std::string formats[] = {[ge::FORMAT_NCHW] = "NCHW", [ge::FORMAT_NHWC] = "NHWC", [ge::FORMAT_ND] = "ND"};
  for (auto format : param.format) {
    if (format > sizeof(formats) / sizeof(std::string)) {
        throw std::runtime_error("Unsupported format");
    }

    j["format"].push_back(formats[format]);
  }
}

void nlohmann::to_json(nlohmann::json &j, const OpProto &proto) {
  j = json{{"op", proto.op},
           {"language", proto.language},
           {"input_desc", proto.input_desc},
           {"output_desc", proto.output_desc}};
}
