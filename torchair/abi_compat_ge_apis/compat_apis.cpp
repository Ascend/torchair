#include "external/graph/types.h"
#include "graph/tensor.h"
#include "graph/utils/type_utils.h"
#include "checker.h"
#include "ge/ge_api.h"
#include "graph/model_serialize.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/detail/model_serialize_imp.h"

#include <unordered_map>

#include "tng_status.h"

namespace tng {
namespace compat {
ge::AscendString GetOpDescName(const ge::OpDescPtr &op_desc) {
  auto *v = op_desc->GetNamePtr();
  return (v == nullptr) ? "" : v;
}

ge::AscendString GetOpDescType(const ge::OpDescPtr &op_desc) {
  auto *v = op_desc->GetTypePtr();
  return (v == nullptr) ? "" : v;
}

std::vector<tng::Placement> GetGraphInputPlacemnts(const ge::proto::GraphDef &graph_def) {
  std::vector<tng::Placement> placements;
  auto &graph_attr = graph_def.attr();
  auto iter = graph_attr.find("_input_placements");
  if (iter != graph_attr.end()) {
    for (const auto &placement : iter->second.list().i()) {
      placements.push_back(static_cast<tng::Placement>(placement));
    }
  }
  return placements;
}

std::vector<ge::DataType> GetGraphOutputDtypes(const ge::proto::GraphDef &graph_def) {
  std::vector<ge::DataType> dtypes;
  auto &graph_attr = graph_def.attr();
  auto iter = graph_attr.find("_output_dtypes");
  if (iter != graph_attr.end()) {
    for (const auto &dtype : iter->second.list().i()) {
      dtypes.push_back(static_cast<ge::DataType>(dtype));
    }
  }
  return dtypes;
}

using Name2Index = std::map<std::string, uint32_t>;
template <typename T>
Name2Index GetDescName2Index(const T &descs) {
  Name2Index name2index;
  for (auto &desc : descs) {
    name2index.emplace(desc.name(), name2index.size());
  }
  return name2index;
}

Status ConvertGraphDefToGraph(ge::proto::GraphDef &graph_def, ge::GraphPtr &graph) {
  ge::ComputeGraphPtr compute_graph = nullptr;
  ge::ModelSerializeImp serializer;
  TNG_ASSERT(serializer.UnserializeGraph(compute_graph, graph_def));
  TNG_ASSERT_NOTNULL(compute_graph, "Failed to get compute graph from model");

  std::unordered_map<std::string, std::pair<Name2Index, Name2Index>> op_to_name2index;
  for (auto &op : graph_def.op()) {
    TNG_ASSERT(
      op_to_name2index
        .emplace(op.name(), std::make_pair(GetDescName2Index(op.input_desc()), GetDescName2Index(op.output_desc())))
        .second,
      "Dumplicated op name: %s", op.name().c_str());
  }

  for (auto &node : compute_graph->GetAllNodes()) {
    TNG_ASSERT_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    TNG_ASSERT_NOTNULL(op_desc);
    auto &io_desc_name2index = op_to_name2index[op_desc->GetName()];
    TNG_ASSERT(op_desc->UpdateInputName(io_desc_name2index.first));
    TNG_ASSERT(op_desc->UpdateOutputName(io_desc_name2index.second));
  }

  graph = ge::GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  TNG_ASSERT_NOTNULL(graph, "Failed to create graph from compute graph");
  return Status::Success();
}

Status GeErrorStatus() { return Status::Error("%s", ge::GEGetErrorMsg().c_str()); }

Status DebugString(const ge::Shape &shape) {
  std::stringstream ss;
  auto dims = shape.GetDims();
  if (dims.empty()) {
    return Status::Error("[]");
  }
  ss << "[";
  size_t index = 0U;
  for (; index < (dims.size() - 1U); index++) {
    ss << dims[index] << ", ";
  }
  ss << dims[index] << "]";
  return Status::Error(ss.str().c_str());
}

Status DebugString(const ge::Tensor &tensor) {
  const auto &desc = tensor.GetTensorDesc();
  std::stringstream ss;
  ss << "ge::Tensor(shape=" << DebugString(desc.GetShape()).GetErrorMessage() << ", dtype='"
     << ge::TypeUtils::DataTypeToSerialString(desc.GetDataType())
     << "', addr=" << static_cast<const void *>(tensor.GetData())
     << ", format=" << ge::TypeUtils::FormatToSerialString(desc.GetFormat()) << ")";
  return Status::Error(ss.str().c_str());
}

ge::AscendString DebugString(const ge::DataType &dtype) { return ge::TypeUtils::DataTypeToSerialString(dtype).c_str(); }
}  // namespace compat
}  // namespace tng
