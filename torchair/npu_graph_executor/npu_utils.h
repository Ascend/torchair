#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_

#include "tng_status.h"

namespace tng {
Status GetCurrentStream(void **stream);
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_