#include "utils.h"

#include "checker.h"
#include "compat_apis.h"
#include "executor.h"
#include "external/graph/types.h"
#include "graph/tensor.h"
#include "graph/utils/type_utils.h"
#include "graph_data.h"
#include "session.h"
#include "torch/torch.h"

#include "ATen/CPUFunctions.h"

namespace tng {
namespace {
std::string DeviceTypeToString(c10::DeviceType device_type) {
  switch (device_type) {
    case c10::DeviceType::PrivateUse1:
      return "PrivateUse1";
    case c10::DeviceType::CPU:
      return "CPU";
    case c10::DeviceType::CUDA:
      return "CUDA";
    case c10::DeviceType::HIP:
      return "HIP";
    case c10::DeviceType::XLA:
      return "XLA";
    case c10::DeviceType::MPS:
      return "MPS";
    case c10::DeviceType::IPU:
      return "IPU";
    case c10::DeviceType::XPU:
      return "XPU";
    case c10::DeviceType::HPU:
      return "HPU";
    case c10::DeviceType::VE:
      return "VE";
    case c10::DeviceType::Lazy:
      return "Lazy";
    case c10::DeviceType::Meta:
      return "Meta";
    case c10::DeviceType::MTIA:
      return "MTIA";
    default:
      return "unsupported DeviceType";
  }
}
}  // namespace

std::string DebugString(const tng::Placement &placement) {
  switch (placement) {
    case tng::Placement::HOST:
      return "Host";
    case tng::Placement::DEVICE:
      return "NPU";
    default:
      return "Unknown";
  }
}

std::string DebugString(const ge::DataType &dtype) {
  return compat::DebugString(dtype).GetString();
}

std::string DebugString(const std::vector<ge::DataType> &dtypes) {
  if (dtypes.empty()) {
    return "[]";
  }
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0U; i < dtypes.size() - 1U; ++i) {
    ss << DebugString(dtypes[i]) << ", ";
  }
  return ss.str() + DebugString(dtypes.back()) + "]";
}

std::string DebugString(const std::vector<tng::Placement> &placements) {
  if (placements.empty()) {
    return "[]";
  }
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0U; i < placements.size() - 1U; ++i) {
    ss << DebugString(placements[i]) << ", ";
  }
  return ss.str() + DebugString(placements.back()) + "]";
}

std::string DebugString(const ge::CompiledGraphSummary &summary) {
  std::stringstream ss;
  ss << "static compiled: " << (summary.IsStatic() ? "True" : "False");
  if (summary.IsStatic()) {
    ss << std::endl;
    size_t workspace_size = 0U;
    (void)summary.GetFeatureMemorySize(workspace_size);
    bool is_workspace_refreshable = false;
    (void)summary.GetFeatureMemoryBaseRefreshable(is_workspace_refreshable);
    size_t const_size = 0U;
    (void)summary.GetConstMemorySize(const_size);
    ss << "workspace size: " << workspace_size << std::endl;
    ss << "workspace refreshable: " << (is_workspace_refreshable ? "True" : "False") << std::endl;
    ss << "const size: " << const_size;
  }
  return ss.str();
}

std::string DebugString(const GraphData &graph_data) {
  std::stringstream ss;
  ss << "Summary of graph id: " << graph_data.id << std::endl;
  if (graph_data.summary != nullptr) {
    ss << DebugString(*graph_data.summary) << std::endl;
  }
  ss << "num nodes: " << graph_data.graph_def.op_size() << std::endl;
  ss << "input placements: " << DebugString(graph_data.input_placements) << std::endl;
  ss << "output dtypes :" << DebugString(graph_data.output_dtypes) << std::endl;
  ss << "executor type :" << ((graph_data.executor_type == tng::ExecutorType::NPU) ? "NPU" : "DEFAULT");
  return ss.str();
}

std::string DebugString(const at::Tensor &tensor) {
  std::stringstream ss;
  ss << "at::Tensor(shape=" << tensor.sizes() << ", dtype='" << tensor.dtype()
     << "', addr=" << tensor.storage().data_ptr().get() << ")";
  return ss.str();
}

std::string DebugString(const c10::optional<at::Tensor> &tensor) {
  if (tensor.has_value()) {
    return DebugString(tensor.value());
  }
  return "None";
}

std::string DebugString(const c10::Device &device) {
  std::stringstream ss;
  ss << DeviceTypeToString(device.type()) << ":" << static_cast<int>(device.index());
  return ss.str();
}

std::string DebugString(const ge::Shape &shape) {
  return compat::DebugString(shape).GetErrorMessage();
}

std::string DebugString(const ge::Tensor &tensor) {
  return compat::DebugString(tensor).GetErrorMessage();
}

Status GePlacementToAtDeviceType(const ge::Placement &ge_placement, c10::DeviceType &device_type) {
  if (ge_placement == ge::Placement::kPlacementHost) {
    device_type = c10::DeviceType::CPU;
  } else if (ge_placement == ge::Placement::kPlacementDevice) {
    device_type = c10::DeviceType::PrivateUse1;
  } else {
    return Status::Error("Unsupported ge placement %d.", ge_placement);
  }
  return Status::Success();
}

Status AtDtypeToGeDtype(const c10::ScalarType &dtype, ge::DataType &ge_dtype) {
#define ATEN2GE_MAP_TYPE(T1, T2) \
  case T1:                       \
    ge_dtype = T2;               \
    return Status::Success()
  switch (dtype) {
    ATEN2GE_MAP_TYPE(c10::ScalarType::Bool, ge::DataType::DT_BOOL);
    ATEN2GE_MAP_TYPE(c10::ScalarType::Byte, ge::DataType::DT_UINT8);
    ATEN2GE_MAP_TYPE(c10::ScalarType::Char, ge::DataType::DT_INT8);
    ATEN2GE_MAP_TYPE(c10::ScalarType::Short, ge::DataType::DT_INT16);
    ATEN2GE_MAP_TYPE(c10::ScalarType::Int, ge::DataType::DT_INT32);
    ATEN2GE_MAP_TYPE(c10::ScalarType::Long, ge::DataType::DT_INT64);
    ATEN2GE_MAP_TYPE(c10::ScalarType::Half, ge::DataType::DT_FLOAT16);
    ATEN2GE_MAP_TYPE(c10::ScalarType::Float, ge::DataType::DT_FLOAT);
    ATEN2GE_MAP_TYPE(c10::ScalarType::Double, ge::DataType::DT_DOUBLE);
    default:
      return Status::Error("Unsupported torch type %d by ge", dtype);
  }
}

Status GeDtypeToAtDtype(const ge::DataType &ge_dtype, c10::ScalarType &dtype) {
#define GE2ATEN_MAP_TYPE(T1, T2) \
  case T1:                       \
    dtype = T2;                  \
    return Status::Success()
  switch (ge_dtype) {
    GE2ATEN_MAP_TYPE(ge::DataType::DT_BOOL, c10::ScalarType::Bool);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_UINT8, c10::ScalarType::Byte);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_INT8, c10::ScalarType::Char);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_INT16, c10::ScalarType::Short);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_INT32, c10::ScalarType::Int);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_INT64, c10::ScalarType::Long);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_FLOAT16, c10::ScalarType::Half);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_FLOAT, c10::ScalarType::Float);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_DOUBLE, c10::ScalarType::Double);
    default:
      return Status::Error("Unsupported ge type %d by torch", ge_dtype);
  }
}

Status AssembleDataToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t *data) {};
  // The input at tensor must be contiguous(), but not necessarily matched.
  // Therefore, when getting data_ptr, the calculation of the data_ptr address needs to skip storage_offset,
  // and the calculation of nbytes needs to be based on the shape after view.
  TNG_ASSERT_GE_OK(
      ge_tensor.SetData(static_cast<uint8_t *>(tensor.data_ptr()), tensor.numel() * tensor.element_size(), kDoNothing));
  return Status::Success();
}

Status AssembleShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  auto desc = ge_tensor.GetTensorDesc();
  desc.SetShape(ge::Shape(tensor.sizes().vec()));
  TNG_ASSERT_GE_OK(ge_tensor.SetTensorDesc(desc));
  return Status::Success();
}

Status AssembleDataAndShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));
  TNG_RETURN_IF_ERROR(AssembleShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AtTensorToGeTensor(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  ge::TensorDesc desc;

  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  TNG_RETURN_IF_ERROR(AtDtypeToGeDtype(tensor.dtype().toScalarType(), ge_dtype));

  desc.SetDataType(ge_dtype);
  desc.SetShape(ge::Shape(tensor.sizes().vec()));
  desc.SetFormat(ge::FORMAT_ND);

  ge_tensor.SetTensorDesc(desc);
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));

  return Status::Success();
}

using RawGeDataPtr = std::unique_ptr<uint8_t[], ge::Tensor::DeleteFunc>;
namespace {
void DeleteGeDataPtr(void *data) {
  if (data != nullptr) {
    delete static_cast<RawGeDataPtr *>(data);
  }
}
}  // namespace

Status GeTensorToAtTensor(ge::Tensor &ge_tensor, at::Tensor &tensor) {
  c10::ScalarType tensor_dtype = c10::ScalarType::Float;
  const ge::TensorDesc &tensor_desc = ge_tensor.GetTensorDesc();
  TNG_RETURN_IF_ERROR(GeDtypeToAtDtype(tensor_desc.GetDataType(), tensor_dtype));
  c10::DeviceType device_type = c10::DeviceType::CPU;
  TNG_RETURN_IF_ERROR(GePlacementToAtDeviceType(tensor_desc.GetPlacement(), device_type));
  at::TensorOptions option = at::TensorOptions().dtype(tensor_dtype).device(device_type);
  tensor = at::cpu::empty({0}, option);

  RawGeDataPtr ge_data_ptr = ge_tensor.ResetData();
  auto raw_ge_data = static_cast<void *>(ge_data_ptr.get());
  auto ge_ctx = std::make_unique<RawGeDataPtr>(std::move(ge_data_ptr));

  static torch::DeleterFnPtr kGeDatatDeleter = &DeleteGeDataPtr;
  at::DataPtr c10_data_ptr(raw_ge_data, ge_ctx.release(), kGeDatatDeleter, tensor.device());

  auto dims = ge_tensor.GetTensorDesc().GetShape().GetDims();
  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(dims, tensor.dtype().itemsize());
  at::Storage storage = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), tensor_nbytes,
                                                              std::move(c10_data_ptr), nullptr, false);
  tensor.set_(storage, 0, dims);
  return Status::Success();
}
}  // namespace tng