#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_UTILS_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_UTILS_H_

#include "external/graph/types.h"
#include "executor.h"
#include "graph_data.h"
#include "graph/ge_tensor.h"
#include "graph/tensor.h"
#include "graph/utils/type_utils.h"
#include "torch/torch.h"
#include "session.h"
#include "checker.h"

namespace tng {
std::string DebugString(const GraphData &graph_data);

std::string DebugString(const at::Tensor &tensor);

std::string DebugString(const ge::Shape &shape);

std::string DebugString(const ge::Tensor &tensor);

std::string DebugString(const c10::optional<at::Tensor> &tensor);

std::string DebugString(const c10::Device &device);

Status GePlacementToAtDeviceType(const ge::Placement &ge_placement, c10::DeviceType &device_type);

Status AtTensorToGeTensor(const at::Tensor &tensor, ge::Tensor &ge_tensor);

Status GeTensorToAtTensor(ge::Tensor &ge_tensor, at::Tensor &tensor);

Status AtDtypeToGeDtype(const c10::ScalarType &dtype, ge::DataType &ge_dtype);

Status AssembleDataToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor);

Status AssembleDimsToGeShape(const at::IntArrayRef &dims, ge::GeShape &ge_shape);

Status AssembleDataAndShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor);

Status GeDtypeToAtDtype(const ge::DataType &ge_dtype, c10::ScalarType &dtype);

Status CloneGraph(const ge::Graph &old_graph, ge::Graph &new_graph);

}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_UTILS_H_