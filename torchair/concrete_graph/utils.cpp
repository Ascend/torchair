#include "utils.h"

#include "checker.h"
#include "compat_apis.h"
#include "executor.h"
#include "graph_data.h"
#include "session.h"

#include "exe_graph/runtime/tensor_data.h"
#include "graph/tensor.h"
#include "graph/types.h"

#include "ATen/CPUFunctions.h"
#include "torch/torch.h"

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
  return compat::DebugString(dtype).GetErrorMessage();
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

std::string DebugString(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return "[]";
  }
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0U; i < shape.size() - 1U; ++i) {
    ss << shape[i] << ", ";
  }
  return ss.str() + std::to_string(shape.back()) + "]";
}

std::string DebugString(const std::vector<std::vector<int64_t>> &shapes) {
  if (shapes.empty()) {
    return "[]";
  }
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0U; i < shapes.size() - 1U; ++i) {
    ss << DebugString(shapes[i]) << ", ";
  }
  return ss.str() + DebugString(shapes.back()) + "]";
}

std::string DebugString(const std::vector<ge::Shape> &shapes) {
  if (shapes.empty()) {
    return "[]";
  }
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0U; i < shapes.size() - 1U; ++i) {
    ss << DebugString(shapes[i].GetDims()) << ", ";
  }
  return ss.str() + DebugString(shapes.back()) + "]";
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
  ss << "input placements: " << DebugString(graph_data.input_placements) << std::endl;
  ss << "hint inputs shape: " << DebugString(graph_data.inputs_shape) << std::endl;
  ss << "hint outputs shape: " << DebugString(graph_data.outputs_shape) << std::endl;
  ss << "output dtypes :" << DebugString(graph_data.output_dtypes) << std::endl;
  ss << "executor type :" << ((graph_data.executor_type == tng::ExecutorType::NPU) ? "NPU" : "DEFAULT");
  return ss.str();
}

std::string DebugString(const at::Tensor &tensor) {
  std::stringstream ss;
  ss << "at::Tensor(shape=" << tensor.sizes() << ", dtype='" << tensor.dtype() << "', device=" << tensor.device()
     << ", addr=" << tensor.storage().data_ptr().get() << ")";
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

std::string DebugString(const gert::Shape &shape) {
  return compat::DebugString(shape).GetErrorMessage();
}

std::string DebugString(const ge::Tensor &tensor) {
  return compat::DebugString(tensor).GetErrorMessage();
}

std::string DebugString(const gert::Tensor &tensor) {
  return compat::DebugString(tensor).GetErrorMessage();
}

std::vector<int64_t> GetDims(const gert::Shape &shape) {
    std::vector<int64_t> dims;
    size_t dim_num = shape.GetDimNum();
    dims.reserve(dim_num);
    for (size_t i = 0U; i < dim_num; ++i) {
      dims.push_back(shape.GetDim(i));
    }
    return dims;
}

std::vector<int64_t> GetGeTensorShape(const ge::Tensor &tensor) {
  size_t rank = tensor.GetShapeDimNum();
  std::vector<int64_t> shape(rank);

  for (size_t i = 0U; i < rank; ++i) {
    shape[i] = tensor.GetShapeDim(i);
  }
  return shape;
}

std::vector<int64_t> GetGeTensorShape(const gert::Tensor &tensor) {
  return GetDims(tensor.GetOriginShape());
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

Status GePlacementToAtDeviceType(const gert::TensorPlacement &ge_placement, c10::DeviceType &device_type) {
  if (ge_placement == gert::TensorPlacement::kOnHost || ge_placement == gert::TensorPlacement::kFollowing) {
    device_type = c10::DeviceType::CPU;
  } else if (ge_placement == gert::TensorPlacement::kOnDeviceHbm ||
             ge_placement == gert::TensorPlacement::kOnDeviceP2p) {
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
    ATEN2GE_MAP_TYPE(c10::ScalarType::BFloat16, ge::DataType::DT_BF16);
    ATEN2GE_MAP_TYPE(c10::ScalarType::ComplexHalf, ge::DataType::DT_COMPLEX32);
    ATEN2GE_MAP_TYPE(c10::ScalarType::ComplexFloat, ge::DataType::DT_COMPLEX64);
    ATEN2GE_MAP_TYPE(c10::ScalarType::ComplexDouble, ge::DataType::DT_COMPLEX128);
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
    GE2ATEN_MAP_TYPE(ge::DataType::DT_BF16, c10::ScalarType::BFloat16);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_COMPLEX32, c10::ScalarType::ComplexHalf);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_COMPLEX64, c10::ScalarType::ComplexFloat);
    GE2ATEN_MAP_TYPE(ge::DataType::DT_COMPLEX128, c10::ScalarType::ComplexDouble);
    default:
      return Status::Error("Unsupported ge type %d by torch", ge_dtype);
  }
}

Status AssembleDataToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor, bool refresh_size) {
  // Performance optimization:
  // When at::tensor address is not updated, there is no need to refresh the ge::tensor memory address again.
  if (ge_tensor.GetData() == static_cast<uint8_t *>(tensor.data_ptr())) {
    return Status::Success();
  }

  const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t *data) {};
  // The input at tensor must be contiguous(), but not necessarily matched.
  // Therefore, when getting data_ptr, the calculation of the data_ptr address needs to skip storage_offset,
  // and the calculation of nbytes needs to be based on the shape after view.
  ge_tensor.ResetData(static_cast<uint8_t *>(tensor.data_ptr()),
                      static_cast<size_t>(tensor.numel() * tensor.element_size()), kDoNothing);
  return Status::Success();
}

Status AssembleDataToGe(const at::Tensor &tensor, gert::Tensor &ge_tensor, bool refresh_size) {
  // Performance optimization:
  // When at::tensor address is not updated, there is no need to refresh the ge::tensor memory address again.
  if (ge_tensor.GetAddr() == tensor.data_ptr()) {
    return Status::Success();
  }

  // The input at tensor must be contiguous(), but not necessarily matched.
  // Therefore, when getting data_ptr, the calculation of the data_ptr address needs to skip storage_offset,
  // and the calculation of nbytes needs to be based on the shape after view.
  ge_tensor.MutableTensorData().SetAddr(tensor.data_ptr(), nullptr);
  if (refresh_size) {
    ge_tensor.MutableTensorData().SetSize(static_cast<size_t>(tensor.numel() * tensor.element_size()));
  }
  return Status::Success();
}

Status AssembleDimsToShape(const at::IntArrayRef &dims, ge::Tensor &ge_tensor) {
  if (ge_tensor.GetShapeDimNum() != dims.size()) {
    TNG_ASSERT_GE_OK(ge_tensor.SetShapeDimNum(dims.size()));
  }
  for (size_t i = 0U; i < dims.size(); ++i) {
    TNG_ASSERT_GE_OK(ge_tensor.SetShapeDim(i, dims[i]));
  }

  return Status::Success();
}

Status AssembleDimsToShape(const at::IntArrayRef &dims, gert::Tensor &ge_tensor) {
  if (ge_tensor.GetOriginShape().GetDimNum() != dims.size()) {
    ge_tensor.MutableOriginShape().SetDimNum(dims.size());
  }
  if (ge_tensor.GetStorageShape().GetDimNum() != dims.size()) {
    ge_tensor.MutableStorageShape().SetDimNum(dims.size());
  }
  for (size_t i = 0U; i < dims.size(); ++i) {
    ge_tensor.MutableOriginShape().SetDim(i, dims[i]);
    ge_tensor.MutableStorageShape().SetDim(i, dims[i]);
  }

  return Status::Success();
}

Status AssembleDataAndShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));

  TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), ge_tensor));
  return Status::Success();
}

Status AssembleDataAndShapeToGe(const at::Tensor &tensor, gert::Tensor &ge_tensor) {
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));

  TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), ge_tensor));
  return Status::Success();
}

Status AtTensorToGeTensor(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  TNG_RETURN_IF_ERROR(AtDtypeToGeDtype(tensor.dtype().toScalarType(), ge_dtype));
  ge_tensor.SetDataType(ge_dtype);
  ge_tensor.SetFormat(ge::FORMAT_ND);
  ge_tensor.SetPlacement(tensor.device().is_privateuseone() ? ge::Placement::kPlacementDevice
                                                            : ge::Placement::kPlacementHost);

  TNG_RETURN_IF_ERROR(AssembleDataAndShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AtTensorToGeTensor(const at::Tensor &tensor, gert::Tensor &ge_tensor) {
  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  TNG_RETURN_IF_ERROR(AtDtypeToGeDtype(tensor.dtype().toScalarType(), ge_dtype));
  ge_tensor.SetDataType(ge_dtype);
  ge_tensor.SetOriginFormat(ge::FORMAT_ND);
  ge_tensor.SetStorageFormat(ge::FORMAT_ND);
  TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), ge_tensor));

  ge_tensor.SetData(gert::TensorData(
      tensor.data_ptr(), nullptr, static_cast<size_t>(tensor.numel() * tensor.element_size()),
      tensor.device().is_privateuseone() ? gert::TensorPlacement::kOnDeviceHbm : gert::TensorPlacement::kOnHost));
  return Status::Success();
}

using RawGeDataPtr = std::unique_ptr<uint8_t[], ge::Tensor::DeleteFunc>;
namespace {
void DeleteGeDataPtr(void *data) {
  if (data != nullptr) {
    delete static_cast<RawGeDataPtr *>(data);
  }
}

void FreeGeBlockPtr(void *data) {
  if (data != nullptr) {
    static_cast<ge::MemBlock *>(data)->Free();
    data = nullptr;
  }
}

struct Context {
  Context(gert::TensorAddrManager mgr, gert::TensorAddress address) {
    manager = mgr;
    addr = address;
  };

  ~Context() {
    manager(addr, gert::kFreeTensor, nullptr);
    manager = nullptr;
    addr = nullptr;
  };

  gert::TensorAddrManager manager;
  gert::TensorAddress addr;
};

void ContextDeleter(void *data) {
  auto ctx = (Context*)data;
  delete ctx;
}
}  // namespace

Status GeTensorToAtTensor(ge::Tensor &ge_tensor, at::Tensor &tensor) {
  c10::ScalarType tensor_dtype = c10::ScalarType::Float;
  TNG_RETURN_IF_ERROR(GeDtypeToAtDtype(ge_tensor.GetDataType(), tensor_dtype));
  c10::DeviceType device_type = c10::DeviceType::CPU;
  TNG_RETURN_IF_ERROR(GePlacementToAtDeviceType(ge_tensor.GetPlacement(), device_type));
  at::TensorOptions option = at::TensorOptions().dtype(tensor_dtype).device(device_type);
  tensor = at::empty({0}, option);

  RawGeDataPtr ge_data_ptr = ge_tensor.ResetData();
  auto raw_ge_data = static_cast<void *>(ge_data_ptr.get());
  auto ge_ctx = std::make_unique<RawGeDataPtr>(std::move(ge_data_ptr));
  static torch::DeleterFnPtr kGeDatatDeleter = &DeleteGeDataPtr;
  at::DataPtr c10_data_ptr(raw_ge_data, ge_ctx.release(), kGeDatatDeleter, tensor.device());

  const std::vector<int64_t> &dims = GetGeTensorShape(ge_tensor);
  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(dims, tensor.dtype().itemsize());

  at::Storage storage;
  if (device_type == c10::DeviceType::PrivateUse1) {
    // get npu storage constructor from register and construct storage
    auto fptr = c10::GetStorageImplCreate(device_type);
    auto allocator = c10::GetAllocator(device_type);
#if defined(TNG_TORCH_VERSION) && (TNG_TORCH_VERSION < 20300)  // v2.3.0
    storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator, true);
#else
    storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator->allocate(0), allocator, true);
#endif
    storage.unsafeGetStorageImpl()->set_nbytes(tensor_nbytes);
    storage.set_data_ptr(std::move(c10_data_ptr));
  } else {
    storage = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), tensor_nbytes,
                                                    std::move(c10_data_ptr), c10::GetAllocator(device_type), true);
  }

  tensor.set_(storage, 0, dims);
  return Status::Success();
}

Status GeTensorToAtTensor(gert::Tensor &ge_tensor, at::Tensor &tensor) {
  c10::ScalarType tensor_dtype = c10::ScalarType::Float;
  TNG_RETURN_IF_ERROR(GeDtypeToAtDtype(ge_tensor.GetDataType(), tensor_dtype));
  c10::DeviceType device_type = c10::DeviceType::CPU;
  TNG_RETURN_IF_ERROR(GePlacementToAtDeviceType(ge_tensor.GetPlacement(), device_type));
  at::TensorOptions option = at::TensorOptions().dtype(tensor_dtype).device(device_type);
  tensor = at::empty({0}, option);

  const std::vector<int64_t> &dims = GetDims(ge_tensor.GetOriginShape());
  if (GetDims(ge_tensor.GetStorageShape()) != dims) {
    return Status::Error("Unsupported ge tensor with different origin shape and storage shape, ",
                         DebugString(ge_tensor).c_str());
  }
  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(dims, tensor.dtype().itemsize());

  gert::TensorAddrManager block_manager = nullptr;
  gert::TensorAddress backup_addr = ge_tensor.GetAddr();
  gert::TensorAddress addr_block = ge_tensor.MutableTensorData().Release(block_manager);
  
  at::DataPtr c10_data_ptr;
  if (block_manager) {
    void *addr = nullptr;
    TNG_ASSERT(block_manager(addr_block, gert::kGetTensorAddress, &addr) == ge::GRAPH_SUCCESS,
               "Get ge tensor addr error.");
    Context *ctx = new Context(block_manager, addr_block);
    c10_data_ptr = at::DataPtr(addr, ctx, &ContextDeleter, tensor.device());
  } else {
    if (device_type == c10::DeviceType::PrivateUse1) {
      c10_data_ptr = at::DataPtr(backup_addr, nullptr, &c10::detail::deleteNothing, tensor.device());
    } else {
      c10_data_ptr = c10::GetAllocator(device_type)->allocate(tensor_nbytes);
      memcpy(c10_data_ptr.get(), addr_block, tensor_nbytes);
    }
  }

  at::Storage storage;
  if (device_type == c10::DeviceType::PrivateUse1) {
    // get npu storage constructor from register and construct storage
    static auto fptr = c10::GetStorageImplCreate(device_type);
    auto allocator = c10::GetAllocator(device_type);
#if defined(TNG_TORCH_VERSION) && (TNG_TORCH_VERSION < 20300)  // v2.3.0
    storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator, true);
#else
    storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator->allocate(0), allocator, true);
#endif
    storage.unsafeGetStorageImpl()->set_nbytes(tensor_nbytes);
    storage.set_data_ptr(std::move(c10_data_ptr));
  } else {
    storage = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), tensor_nbytes,
                                                    std::move(c10_data_ptr), c10::GetAllocator(device_type), true);
  }

  tensor.set_(storage, 0, dims);
  return Status::Success();
}

std::vector<bool> Split(const std::string &str, char pattern) {
  std::string str_pattern(1, pattern);
  std::vector<bool> res_vec;
  if (str.empty()) {
    return res_vec;
  }
  std::string str_and_pattern = str + str_pattern;
  size_t pos = str_and_pattern.find(str_pattern);
  size_t size = str_and_pattern.size();
  while (pos != std::string::npos) {
    std::string sub_str = str_and_pattern.substr(0, pos);
    bool sub_bool;
    std::istringstream(sub_str)>>sub_bool;
    res_vec.push_back(sub_bool);
    str_and_pattern = str_and_pattern.substr(pos + str_pattern.size(), size);
    pos = str_and_pattern.find(str_pattern);
  }
  return res_vec;
}
}  // namespace tng

