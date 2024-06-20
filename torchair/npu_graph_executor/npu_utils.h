#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_

#include "tng_status.h"
#include "torch/torch.h"
#include "graph/tensor.h"

namespace tng {
Status GetCurrentStream(void **stream);

Status AssembleDimsToOriginShape(const at::IntArrayRef &dims, ge::Tensor &ge_tensor);

Status AssembleStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor);

Status AssembleDataAndStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor);

Status AtNpuTensorToGeTensor(const at::Tensor &tensor, ge::Tensor &ge_tensor);

bool IsBaseFormat(const ge::Format &format);

}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_