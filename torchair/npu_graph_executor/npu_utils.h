#ifndef TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_
#define TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/tensor_data.h"
#include "graph/tensor.h"
#include "tng_status.h"
#include "torch/torch.h"
#include "ge/ge_allocator.h"

namespace tng {
Status GetCurrentStream(void **stream);

Status H2DMemcpy(void *dst, size_t destMax, const void *src, size_t count, void *stream);

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
                            const std::vector<const at::Tensor*> &torch_inputs,
                            std::map<ge::AscendString, ge::AscendString> &load_options);

Status AssembleHostInputOption(const std::vector<const at::Tensor*> &torch_inputs,
                               std::map<ge::AscendString, ge::AscendString> &load_options);

Status GetShapeFromGeTensor(std::vector<int64_t> &real_output_shape, const ge::Tensor &ge_tensor);

Status GetShapeFromGeTensor(std::vector<int64_t> &real_output_shape, const gert::Tensor &ge_tensor);

at::Tensor MakeAtTensor(const std::vector<int64_t> &dims, c10::ScalarType &torch_dtype, size_t tensor_nbytes,
                        at::DataPtr&& data_ptr);

Status UpdateTensorInfos(ge::Tensor &ge_tensor, const std::vector<int64_t> &shape, const ge::Format format,
                         const ge::DataType data_type);

Status UpdateTensorInfos(gert::Tensor &ge_tensor, const std::vector<int64_t> &shape, const ge::Format format,
                         const ge::DataType data_type);

Status UpdateTensorData(ge::Tensor &ge_tensor, void *addr, const size_t data_size);

Status UpdateTensorData(gert::Tensor &ge_tensor, void *addr, const size_t data_size);

bool CheckCANNVersion82RC1();

bool IsSupportHostInput();
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_NPU_GRAPH_EXECUTOR_NPU_UTILS_H_
