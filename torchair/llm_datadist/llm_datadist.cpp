#include <memory>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "llm_datadist.h"

namespace {
constexpr uint32_t kSuccess = 0;
constexpr uint32_t kFailure = 1;
}  // namespace

namespace llm_datadist {
// llm datadist
c10::ScalarType ToScalarType(const TorchDataType &ge_dtype) {
  static const std::map<TorchDataType, c10::ScalarType> kDtypeToScalarType = {
      {TorchDataType::kBool, c10::ScalarType::Bool},
      {TorchDataType::kUint8, c10::ScalarType::Byte},
      {TorchDataType::kInt8, c10::ScalarType::Char},
      {TorchDataType::kInt16, c10::ScalarType::Short},
      {TorchDataType::kInt32, c10::ScalarType::Int},
      {TorchDataType::kInt64, c10::ScalarType::Long},
      {TorchDataType::kBfloat16, c10::ScalarType::BFloat16},
      {TorchDataType::kFloat16, c10::ScalarType::Half},
      {TorchDataType::kFloat32, c10::ScalarType::Float},
      {TorchDataType::kFloat64, c10::ScalarType::Double},
      {TorchDataType::kComplex32, c10::ScalarType::ComplexHalf},
      {TorchDataType::kComplex64, c10::ScalarType::ComplexFloat},
      {TorchDataType::kComplex128, c10::ScalarType::ComplexDouble},
  };
  c10::ScalarType scalar_type = c10::ScalarType::Undefined;
  const auto &it = kDtypeToScalarType.find(ge_dtype);
  if (it != kDtypeToScalarType.cend()) {
    scalar_type = it->second;
  }
  return scalar_type;
}

std::pair<uint32_t, std::vector<at::Tensor>> AsTorchTensor(const std::vector<int64_t> &dims, const int32_t ge_data_type,
                                                           const std::vector<uintptr_t> &addresses) {
  std::vector<at::Tensor> at_tensors;
  c10::ScalarType tensor_dtype = ToScalarType(static_cast<TorchDataType>(ge_data_type));
  if (tensor_dtype == c10::ScalarType::Undefined) {
    return {kFailure, at_tensors};
  }
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  at::TensorOptions option = at::TensorOptions().dtype(tensor_dtype).device(device_type);

  at_tensors.reserve(addresses.size());
  for (auto dev_addr : addresses) {
    auto tensor = at::empty({0}, option);
    auto address = reinterpret_cast<void *>(dev_addr);
    at::DataPtr c10_data_ptr(address, address, [](void *) {}, tensor.device());

    size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(dims, tensor.dtype().itemsize());
    at::Storage storage;
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

    tensor.set_(storage, 0, dims);
    at_tensors.emplace_back(std::move(tensor));
  }
  return {kSuccess, at_tensors};
}
}  // namespace llm_datadist
