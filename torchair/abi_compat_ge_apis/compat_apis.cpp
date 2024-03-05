#include <unordered_map>

#include "checker.h"
#include "external/graph/types.h"
#include "ge/ge_api.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/model_serialize.h"
#include "graph/tensor.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/type_utils.h"

#include "tng_status.h"

namespace tng {
namespace compat {
using Name2Index = std::map<std::string, uint32_t>;
template <typename T>
Name2Index GetDescName2Index(const T &descs) {
  Name2Index name2index;
  for (auto &desc : descs) {
    name2index.emplace(desc.name(), name2index.size());
  }
  return name2index;
}

Status ParseGraphFromArray(const void *serialized_proto, size_t proto_size, ge::GraphPtr &graph) {
  TNG_ASSERT_NOTNULL(serialized_proto, "Given serialized proto is nullptr.");
  if (graph == nullptr) {
    graph = std::make_shared<ge::Graph>();
  }
  TNG_ASSERT(graph->LoadFromSerializedModelArray(serialized_proto, proto_size) == ge::GRAPH_SUCCESS);
  return Status::Success();
}

Status GeErrorStatus() {
  return Status::Error("%s", ge::GEGetErrorMsg().c_str());
}

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
     << "', device=" << (desc.GetPlacement() == ge::Placement::kPlacementHost ? "CPU" : "NPU")
     << ", addr=" << static_cast<const void *>(tensor.GetData())
     << ", format=" << ge::TypeUtils::FormatToSerialString(desc.GetFormat()) << ")";
  return Status::Error(ss.str().c_str());
}

ge::AscendString DebugString(const ge::DataType &dtype) {
  return ge::TypeUtils::DataTypeToSerialString(dtype).c_str();
}
}  // namespace compat
}  // namespace tng
