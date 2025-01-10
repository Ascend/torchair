#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_ABI_COMPAT_GE_COMPAT_APIS_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_ABI_COMPAT_GE_COMPAT_APIS_H_

#include <cstdarg>
#include <cstdio>
#include <vector>
#include "securec.h"

#include "graph_data.h"
#include "tng_status.h"

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tensor.h"

namespace tng {
namespace compat {
Status GeErrorStatus();
Status DebugString(const ge::Shape &shape);
Status DebugString(const gert::Shape &shape);
Status DebugString(const ge::Tensor &tensor);
Status DebugString(const gert::Tensor &tensor);
Status DebugString(const ge::DataType &dtype);
Status DebugString(const ge::Format &format);
Status ParseGraphFromArray(const void *serialized_proto, size_t proto_size, ge::GraphPtr &graph);
}  // namespace compat
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_ABI_COMPAT_GE_COMPAT_APIS_H_