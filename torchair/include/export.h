#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_EXPORT_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_EXPORT_H_

#include <map>

#include "tng_status.h"
#include "external/graph/types.h"
#include "external/graph/ascend_string.h"

namespace tng {
namespace ep {
Status Export(const void *serialized_proto, size_t proto_size,
              const std::map<ge::AscendString, ge::AscendString> &options);
}  // namespace ep
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_EXPORT_H_
