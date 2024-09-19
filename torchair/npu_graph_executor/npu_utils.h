#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/tensor_data.h"
#include "graph/tensor.h"
#include "tng_status.h"
#include "torch/torch.h"

namespace tng {
Status GetCurrentStream(void **stream);

Status AssembleDimsToOriginShape(const at::IntArrayRef &dims, ge::Tensor &ge_tensor);

Status AssembleStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor);

Status AssembleStorageShapeToGe(const at::Tensor &tensor, gert::Tensor &ge_tensor);

Status AssembleDataAndStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor);

Status AssembleDataAndStorageShapeToGe(const at::Tensor &tensor, gert::Tensor &ge_tensor);

Status AtNpuTensorToGeTensor(const at::Tensor &tensor, ge::Tensor &ge_tensor);

Status AtNpuTensorToGeTensor(const at::Tensor &tensor, gert::Tensor &ge_tensor);

bool IsBaseFormat(const ge::Format &format);

Status AssembleDimsToShape(const at::IntArrayRef &origin_dims, const at::IntArrayRef &storage_dims,
                           gert::Tensor &ge_tensor);

Status AssembleFrozenOption(const std::vector<bool> &frozen_input_flag_list,
                            const std::vector <at::Tensor> &torch_inputs,
                            std::string &frozen_option_value);

}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_