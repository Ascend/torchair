#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_UTILS_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_UTILS_H_

#include "checker.h"
#include "executor.h"
#include "graph_data.h"
#include "session.h"

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/tensor_data.h"
#include "graph/tensor.h"
#include "graph/types.h"

#include "torch/torch.h"

namespace tng {
const char *const OPTION_EXEC_FROZEN_INPUT_INDEXES = "ge.exec.frozenInputIndexes";
const char *const OPTION_EXEC_HOST_INPUT_INDEXES = "ge.exec.hostInputIndexes";

std::string DebugString(const GraphData &graph_data);

std::string DebugString(const at::Tensor &tensor);

std::string DebugString(const ge::Shape &shape);

std::string DebugString(const gert::Shape &shape);

std::string DebugString(const ge::Tensor &tensor);

std::string DebugString(const gert::Tensor &tensor);

std::string DebugString(const c10::optional<at::Tensor> &tensor);

std::string DebugString(const c10::Device &device);

std::string DebugString(const std::vector<std::vector<int64_t>> &shapes);

std::string DebugString(const std::vector<int64_t> &shape);

std::string DebugString(const std::vector<ge::Shape> &shapes);

std::string DebugString(const std::vector<ge::DataType> &dtypes);

std::string DebugString(const ge::DataType &dtype);

std::vector<bool> Split(const std::string &str, char pattern);

std::vector<int64_t> GetGeTensorShape(const ge::Tensor &tensor);

Status GePlacementToAtDeviceType(const ge::Placement &ge_placement, c10::DeviceType &device_type);

Status GePlacementToAtDeviceType(const gert::TensorPlacement &ge_placement, c10::DeviceType &device_type);

Status AtTensorToGeTensor(const at::Tensor &tensor, ge::Tensor &ge_tensor);

Status AtTensorToGeTensor(const at::Tensor &tensor, gert::Tensor &ge_tensor);

Status GeTensorToAtTensor(ge::Tensor &ge_tensor, at::Tensor &tensor);

Status GeTensorToAtTensor(gert::Tensor &ge_tensor, at::Tensor &tensor);

Status AtDtypeToGeDtype(const c10::ScalarType &dtype, ge::DataType &ge_dtype);

Status AssembleDataToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor, bool refresh_size = true);

Status AssembleDataToGe(const at::Tensor &tensor, gert::Tensor &ge_tensor, bool refresh_size = true);

Status AssembleDimsToShape(const at::IntArrayRef &dims, ge::Tensor &ge_tensor);

Status AssembleDimsToShape(const at::IntArrayRef &dims, gert::Tensor &ge_tensor);

Status AssembleDataAndShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor);

Status AssembleDataAndShapeToGe(const at::Tensor &tensor, gert::Tensor &ge_tensor);

Status GeDtypeToAtDtype(const ge::DataType &ge_dtype, c10::ScalarType &dtype);

std::vector<int64_t> GetDims(const gert::Shape &shape);
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_UTILS_H_