#include <unordered_set>
#include <sstream>
#include "graph/operator_factory.h"

namespace cann_ir_ability {

std::string CheckCannCompat(const std::string &optype, const std::vector<std::string> &optional_inputs,
                            const std::vector<std::string> &optional_attrs) noexcept {
  bool is_exist = ge::OperatorFactory::IsExistOp(optype.c_str());
  if (!is_exist) {
    return "OperatorFactory find optype " + optype + " failed, maybe you need upgrade cann version.";
  }

  std::map<ge::AscendString, ge::AscendString> attr_names_types;
  auto op = ge::OperatorFactory::CreateOperator("op_name", optype.c_str());
  auto res = op.GetAllIrAttrNamesAndTypes(attr_names_types);
  if (res != ge::GRAPH_SUCCESS) {
    return "OperatorFactory op " + optype + " , GetAllIrAttrNamesAndTypes failed.";
  }

  std::unordered_set<std::string> support_attrs;
  for (const auto &item : attr_names_types) {
    support_attrs.insert(item.first.GetString());
  }

  std::ostringstream unsupport_optional_input;
  ge::TensorDesc td;
  for (size_t i = 0U; i < optional_inputs.size(); ++i) {
    if (op.UpdateInputDesc(optional_inputs[i].c_str(), td) != ge::GRAPH_SUCCESS) {
      unsupport_optional_input << optional_inputs[i] << (i == (optional_inputs.size() - 1U) ? "" : ", ");
    }
  }
  std::ostringstream unsupport_optional_attr;
  for (size_t i = 0U; i < optional_attrs.size(); ++i) {
    if (support_attrs.find(optional_attrs[i]) == support_attrs.end()) {
      unsupport_optional_attr << optional_attrs[i] << (i == (optional_attrs.size() - 1U) ? "" : ", ");
    }
  }

  if (unsupport_optional_input.str().empty() && unsupport_optional_attr.str().empty()) {
    return "";
  }

  std::ostringstream error_stream;
  error_stream << "optype " << optype << " unsupport optional input [";
  if (!unsupport_optional_input.str().empty()) {
    error_stream << unsupport_optional_input.str();
  }
  error_stream << "], optional attr [";
  if (!unsupport_optional_attr.str().empty()) {
    error_stream << unsupport_optional_attr.str();
  }
  error_stream << "], please upgrade cann version.";
  return error_stream.str();
}
}  // namespace cann_ir_ability
