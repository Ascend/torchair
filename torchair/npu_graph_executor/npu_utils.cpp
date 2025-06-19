#include "npu_utils.h"
#include "checker.h"
#include "graph_data.h"
#include "logger.h"
#include "utils.h"
#include "regex"

#include "torch_npu/inc/core/NPUFormat.h"
#include "torch_npu/inc/core/NPUStream.h"
#include "torch_npu/inc/core/GetCANNInfo.h"

#include "acl/acl_rt.h"
#include "external/graph/types.h"

namespace tng {
constexpr size_t kVersionIndex1 = 1;
constexpr size_t kVersionIndex2 = 2;
constexpr size_t kVersionIndex3 = 3;
constexpr size_t kVersionIndex4 = 4;

static constexpr size_t kFormatNchwRank = 4;
static constexpr size_t kFormatNcdhwRank = 5;

namespace {
ge::Format GetOriginFormatFromAtTensor(const at::Tensor &at_tensor) {
  size_t dim_num = at_tensor.sizes().size();
  ge::Format format = ge::FORMAT_ND;
  switch (dim_num) {
    case kFormatNchwRank:
      format = ge::FORMAT_NCHW;
      break;
    case kFormatNcdhwRank:
      format = ge::FORMAT_NCDHW;
      break;
    default:
      format = ge::FORMAT_ND;
  }

  return format;
}
}  // namespace

bool IsBaseFormat(const ge::Format &format) {
  return (format == ge::FORMAT_ND) || (format == ge::FORMAT_NCHW) || (format == ge::FORMAT_NHWC) ||
         (format == ge::FORMAT_NCDHW);
}

Status GetCurrentStream(void **stream) {
  *stream = c10_npu::getCurrentNPUStream().stream();
  return Status::Success();
}

Status H2DMemcpy(void *dst, size_t destMax, const void *src, size_t count, void *stream) {
  auto stream_ret = aclrtSynchronizeStream(stream);
  TNG_ASSERT(stream_ret == ACL_ERROR_NONE, "ACL sync stream failed, return %d", stream_ret);
  auto ret = aclrtMemcpy(dst, destMax, src, count, ACL_MEMCPY_HOST_TO_DEVICE);
  TNG_ASSERT(ret == ACL_ERROR_NONE, "ACL memory copy failed, return %d", ret);
  return Status::Success();
}

Status AssembleDimsToOriginShape(const at::IntArrayRef &dims, ge::Tensor &ge_tensor) {
  if (ge_tensor.GetOriginShapeDimNum() != dims.size()) {
    TNG_ASSERT_GE_OK(ge_tensor.SetOriginShapeDimNum(dims.size()));
  }
  for (size_t i = 0U; i < dims.size(); ++i) {
    TNG_ASSERT_GE_OK(ge_tensor.SetOriginShapeDim(i, dims[i]));
  }

  return Status::Success();
}

Status AssembleStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  if (tensor.device().is_privateuseone()) {
    const bool is_base_format = IsBaseFormat(ge_tensor.GetFormat());
    TNG_ASSERT(is_base_format || (tensor.storage_offset() == 0),
               "Invalid at::tensor with internal format and offset is %lld.", tensor.storage_offset());

    if (is_base_format) {
      TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), ge_tensor));
    } else {
      TNG_RETURN_IF_ERROR(AssembleDimsToShape(at_npu::native::get_npu_storage_sizes(tensor), ge_tensor));
    }
    TNG_RETURN_IF_ERROR(AssembleDimsToOriginShape(tensor.sizes(), ge_tensor));
  } else {
    TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), ge_tensor));
  }

  return Status::Success();
}

Status AssembleStorageShapeToGe(const at::Tensor &tensor, gert::Tensor &ge_tensor) {
  if (tensor.device().is_privateuseone()) {
    const bool is_base_format = IsBaseFormat(ge_tensor.GetStorageFormat());
    TNG_ASSERT(is_base_format || (tensor.storage_offset() == 0),
               "Invalid at::tensor with internal format and offset is %lld.", tensor.storage_offset());

    if (!is_base_format) {
      const std::vector<int64_t> &storage_sizes = at_npu::native::get_npu_storage_sizes(tensor);
      TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), storage_sizes, ge_tensor));
      const int64_t num =
          std::accumulate(storage_sizes.cbegin(), storage_sizes.cend(), 1LL, std::multiplies<int64_t>());
      ge_tensor.MutableTensorData().SetSize(static_cast<size_t>(num * tensor.element_size()));
      return Status::Success();
    }
  }

  TNG_RETURN_IF_ERROR(AssembleDimsToShape(tensor.sizes(), ge_tensor));
  ge_tensor.MutableTensorData().SetSize(static_cast<size_t>(tensor.numel() * tensor.element_size()));
  return Status::Success();
}

Status AssembleDataAndStorageShapeToGe(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));
  TNG_RETURN_IF_ERROR(AssembleStorageShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AssembleDataAndStorageShapeToGe(const at::Tensor &tensor, gert::Tensor &ge_tensor) {
  TNG_RETURN_IF_ERROR(AssembleDataToGe(tensor, ge_tensor));
  TNG_RETURN_IF_ERROR(AssembleStorageShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AtNpuTensorToGeTensor(const at::Tensor &tensor, ge::Tensor &ge_tensor) {
  ge::TensorDesc desc;

  if (tensor.device().is_privateuseone()) {
    desc.SetOriginFormat(GetOriginFormatFromAtTensor(tensor));
    desc.SetOriginShape(ge::Shape(tensor.sizes().vec()));
    TNG_LOG(DEBUG) << "Set ge tensor origin shape: " << tensor.sizes() << ", from npu at::tensor shape.";

    desc.SetPlacement(ge::Placement::kPlacementDevice);
    desc.SetFormat(ge::Format(at_npu::native::get_npu_format(tensor)));
  } else {
    desc.SetPlacement(ge::Placement::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
  }

  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  TNG_RETURN_IF_ERROR(AtDtypeToGeDtype(tensor.dtype().toScalarType(), ge_dtype));
  desc.SetDataType(ge_dtype);

  TNG_ASSERT_GE_OK(ge_tensor.SetTensorDesc(desc));
  TNG_RETURN_IF_ERROR(AssembleDataAndStorageShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AtNpuTensorToGeTensor(const at::Tensor &tensor, gert::Tensor &ge_tensor) {
  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  TNG_RETURN_IF_ERROR(AtDtypeToGeDtype(tensor.dtype().toScalarType(), ge_dtype));
  ge_tensor.SetDataType(ge_dtype);

  if (tensor.device().is_privateuseone()) {
    ge_tensor.SetPlacement(gert::TensorPlacement::kOnDeviceHbm);
    ge_tensor.SetOriginFormat(GetOriginFormatFromAtTensor(tensor));
    ge_tensor.SetStorageFormat(ge::Format(at_npu::native::get_npu_format(tensor)));
  } else {
    ge_tensor.SetPlacement(gert::TensorPlacement::kOnHost);
    ge_tensor.SetOriginFormat(ge::FORMAT_ND);
    ge_tensor.SetStorageFormat(ge::FORMAT_ND);
  }
  TNG_RETURN_IF_ERROR(AssembleDataAndStorageShapeToGe(tensor, ge_tensor));
  return Status::Success();
}

Status AssembleDimsToShape(const at::IntArrayRef &origin_dims, const at::IntArrayRef &storage_dims,
                           gert::Tensor &ge_tensor) {
  if (ge_tensor.GetOriginShape().GetDimNum() != origin_dims.size()) {
    ge_tensor.MutableOriginShape().SetDimNum(origin_dims.size());
  }
  if (ge_tensor.GetStorageShape().GetDimNum() != storage_dims.size()) {
    ge_tensor.MutableStorageShape().SetDimNum(storage_dims.size());
  }
  for (size_t i = 0U; i < origin_dims.size(); ++i) {
    ge_tensor.MutableOriginShape().SetDim(i, origin_dims[i]);
  }
  for (size_t i = 0U; i < storage_dims.size(); ++i) {
    ge_tensor.MutableStorageShape().SetDim(i, storage_dims[i]);
  }

  return Status::Success();
}

Status AssembleFrozenOption(const std::vector <bool> &frozen_input_flag_list,
                            const std::vector<const at::Tensor*> &torch_inputs,
                            std::map<ge::AscendString, ge::AscendString> &load_options) {
  if (frozen_input_flag_list.empty()) {
    return Status::Success();
  }
  TNG_ASSERT(frozen_input_flag_list.size() == torch_inputs.size());
  std::stringstream frozen_input_flag_list_stream;
  for (size_t i = 0U; i < frozen_input_flag_list.size(); i++) {
    if (frozen_input_flag_list[i] && (*torch_inputs[i]).device().is_privateuseone()) {
      if (frozen_input_flag_list_stream.str() != "") {
        frozen_input_flag_list_stream << ";";
      }
      const std::vector<int64_t> &storage_sizes = at_npu::native::get_npu_storage_sizes(*torch_inputs[i]);
      const int64_t num = std::accumulate(storage_sizes.cbegin(), storage_sizes.cend(), 1LL,
                                          std::multiplies<int64_t>());
      frozen_input_flag_list_stream << i << "," << reinterpret_cast<uintptr_t>(((*torch_inputs[i])).data_ptr()) << ","
                                    << static_cast<size_t>(num * (*torch_inputs[i]).element_size());
    }
  }

  auto frozen_option_value = frozen_input_flag_list_stream.str();
  if (!frozen_option_value.empty()) {
    load_options.insert(std::make_pair(OPTION_EXEC_FROZEN_INPUT_INDEXES, frozen_option_value.c_str()));
  }
  return Status::Success();
}

Status AssembleHostInputOption(const std::vector<const at::Tensor*> &torch_inputs,
                               std::map<ge::AscendString, ge::AscendString> &load_options) {
    if (!IsSupportHostInput()) {
        return Status::Success();
    }
    std::stringstream ss;
    for (size_t i = 0u; i < torch_inputs.size(); ++i) {
        if (!torch_inputs[i]->is_cpu()) {
            continue;
        }
        ss << i << ";";
    }
    auto option = ss.str();
    if (!option.empty()) {
        option.pop_back();
        load_options.insert(std::make_pair(OPTION_EXEC_HOST_INPUT_INDEXES, option.c_str()));
    }
    return Status::Success();
}

Status GetShapeFromGeTensor(std::vector<int64_t> &real_output_shape, const ge::Tensor &ge_tensor) {
  TNG_ASSERT_EQ(real_output_shape.size(), ge_tensor.GetShapeDimNum());
  for (size_t i = 0U; i < real_output_shape.size(); ++i) {
    real_output_shape[i] = ge_tensor.GetShapeDim(i);
  }
  return Status::Success();
}

Status GetShapeFromGeTensor(std::vector<int64_t> &real_output_shape, const gert::Tensor &ge_tensor) {
  TNG_ASSERT_EQ(real_output_shape.size(), ge_tensor.GetStorageShape().GetDimNum());
  for (size_t i = 0U; i < real_output_shape.size(); ++i) {
    real_output_shape[i] = ge_tensor.GetStorageShape().GetDim(i);
  }
  return Status::Success();
}

void FreeMemBlock(void *data) {
  if (data != nullptr) {
    static_cast<ge::MemBlock *>(data)->Free();
    data = nullptr;
  }
}

at::Tensor MakeAtTensor(const std::vector<int64_t> &dims, c10::ScalarType &torch_dtype, size_t tensor_nbytes,
                        at::DataPtr&& data_ptr) {
  at::TensorOptions option = at::TensorOptions().dtype(torch_dtype).device(c10::DeviceType::PrivateUse1);
  at::Tensor tensor = at::empty({0}, option);

  at::Storage storage;
  // get npu storage constructor from register and construct storage
  auto fptr = c10::GetStorageImplCreate(c10::DeviceType::PrivateUse1);
  auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
#if defined(TNG_TORCH_VERSION) && (TNG_TORCH_VERSION < 20300)  // v2.3.0
  storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator, true);
#else
  storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator->allocate(0), allocator, true);
#endif
  storage.unsafeGetStorageImpl()->set_nbytes(tensor_nbytes);
  storage.set_data_ptr(std::move(data_ptr));
  tensor.set_(storage, 0, dims);
  return tensor;
}

Status UpdateTensorInfos(ge::Tensor &ge_tensor, const std::vector<int64_t> &shape, const ge::Format format,
                         const ge::DataType data_type) {
  TNG_ASSERT_GE_OK(ge_tensor.SetDataType(data_type));
  TNG_ASSERT_GE_OK(ge_tensor.SetPlacement(ge::Placement::kPlacementDevice));
  TNG_ASSERT_GE_OK(ge_tensor.SetFormat(format));
  TNG_ASSERT_GE_OK(ge_tensor.SetShapeDimNum(shape.size()));
  for (size_t index = 0U; index < shape.size(); ++index) {
    TNG_ASSERT_GE_OK(ge_tensor.SetShapeDim(index, shape[index]));
  }
  return Status::Success();
}

Status UpdateTensorInfos(gert::Tensor &ge_tensor, const std::vector<int64_t> &shape, const ge::Format format,
                         const ge::DataType data_type) {
  ge_tensor.SetDataType(data_type);
  ge_tensor.SetPlacement(gert::TensorPlacement::kOnDeviceHbm);
  ge_tensor.SetOriginFormat(format);
  ge_tensor.SetStorageFormat(format);
  TNG_RETURN_IF_ERROR(AssembleDimsToShape(shape, ge_tensor));
  return Status::Success();
}

Status UpdateTensorData(ge::Tensor &ge_tensor, void *addr, const size_t data_size) {
  const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t *data) {};
  TNG_ASSERT_GE_OK(ge_tensor.ResetData(static_cast<uint8_t *>(addr), static_cast<size_t>(data_size), kDoNothing));
  return Status::Success();
}

Status UpdateTensorData(gert::Tensor &ge_tensor, void *addr, const size_t data_size) {
  TNG_ASSERT_GE_OK(ge_tensor.MutableTensorData().SetAddr(addr, nullptr));
  ge_tensor.MutableTensorData().SetSize(data_size);
  return Status::Success();
}

int64_t VersionToNum(std::string versionStr)
{
  std::smatch results;
  int64_t major = -1L;
  int64_t minor = -1L;
  int64_t release = -1L;
  int64_t RCVersion = -51L;
  int64_t TVersion = -1L;
  int64_t alphaVersion = 0L;
  if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+)"))) {
    major = stoll(results[kVersionIndex1]);
    minor = stoll(results[kVersionIndex2]);
    RCVersion = stoll(results[kVersionIndex3]);
  } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).([0-9]+)"))) {
    major = stoll(results[kVersionIndex1]);
    minor = stoll(results[kVersionIndex2]);
    release = stoll(results[kVersionIndex3]);
  } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).T([0-9]+)"))) {
    major = stoll(results[kVersionIndex1]);
    minor = stoll(results[kVersionIndex2]);
    TVersion = stoll(results[kVersionIndex3]);
  } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+).alpha([0-9]+)"))) {
    major = stoll(results[kVersionIndex1]);
    minor = stoll(results[kVersionIndex2]);
    RCVersion = stoll(results[kVersionIndex3]);
    alphaVersion = stoll(results[kVersionIndex4]);
  } else {
    return 0L;
  }

  int64_t num = ((major + 1) * 100000000) +
                ((minor + 1) * 1000000) +
                ((release + 1) * 10000) +
                ((RCVersion + 1) * 100 + 5000) +
                ((TVersion + 1) * 100) - (100 - alphaVersion);
  return num;
}

bool CheckCANNVersion(std::string version) {
  std::string currentVersion = GetCANNVersion();
  int64_t currentNum = VersionToNum(currentVersion);
  int64_t boundaryNum = VersionToNum(version);
  return currentNum >= boundaryNum;
}

bool CheckCANNVersion82RC1() {
  const static bool equalOrGreater = []() {
    return CheckCANNVersion("8.2.RC1");
  }();
  return equalOrGreater;
}

bool IsSupportHostInput() {
    // check static shape graph supports channel-associated copy of host input
    return CheckCANNVersion82RC1();
}

}  // namespace tng
