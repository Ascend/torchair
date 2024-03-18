#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_ABI_COMPAT_GE_COMPAT_APIS_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_ABI_COMPAT_GE_COMPAT_APIS_H_

#include <vector>
#include <cstdarg>
#include <cstdio>
#include "securec.h"

#include "tng_status.h"
#include "graph_data.h"

namespace tng {
namespace compat {
Status GeErrorStatus();
Status DebugString(const ge::Shape &shape);
Status DebugString(const ge::Tensor &tensor);
std::string DebugString(const ge::DataType &dtype);
std::string DebugString(const ge::Format &format);
Status ParseGraphFromArray(const void *serialized_proto, size_t proto_size, ge::GraphPtr &graph);
}  // namespace compat
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_ABI_COMPAT_GE_COMPAT_APIS_H_