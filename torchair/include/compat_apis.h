#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_ABI_COMPAT_GE_COMPAT_APIS_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_ABI_COMPAT_GE_COMPAT_APIS_H_

#include <vector>
#include <cstdarg>
#include <cstdio>
#include "securec.h"

#include "tng_status.h"
#include "graph/ge_tensor.h"
#include "graph_data.h"

namespace tng {
namespace compat {
Status GeErrorStatus();
Status DebugString(const ge::Shape &shape);
Status DebugString(const ge::Tensor &tensor);
ge::AscendString GetOpDescName(const ge::OpDescPtr &op_desc);
ge::AscendString GetOpDescType(const ge::OpDescPtr &op_desc);
ge::AscendString DebugString(const ge::DataType &dtype);
Status ConvertGraphDefToGraph(ge::proto::GraphDef &graph_def, ge::GraphPtr &graph);
std::vector<tng::Placement> GetGraphInputPlacemnts(const ge::proto::GraphDef &graph_def);
std::vector<ge::DataType> GetGraphOutputDtypes(const ge::proto::GraphDef &graph_def);
}  // namespace compat
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_ABI_COMPAT_GE_COMPAT_APIS_H_