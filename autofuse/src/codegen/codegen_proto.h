#ifndef __CODEGEN_PROTO_H__
#define __CODEGEN_PROTO_H__

#include "nlohmann/json.hpp"
#include "graph/types.h"
#include "ascir.h"

namespace codegen {
class OpParamDesc {
 public:
  std::string name;
  std::string param_type;
  std::vector<ge::DataType> type;
  std::vector<ge::Format> format;

  OpParamDesc(const std::string &name, const std::string &param_type, const std::vector<ge::DataType> &type,
              const std::vector<ge::Format> &format);
};

class OpProto {
 public:
  std::string op;
  std::string language;
  std::vector<OpParamDesc> input_desc;
  std::vector<OpParamDesc> output_desc;

  static OpProto FromGraph(const ascir::HintGraph& graph);
};

void to_json(nlohmann::json& j, const OpParamDesc& p);
void to_json(nlohmann::json& j, const OpProto& p);
};

#endif
