#include <unordered_map>

#include "checker.h"
#include "tng_status.h"
#include "utils.h"

#include "ge/ge_api.h"
#include "graph/tensor.h"
#include "graph/types.h"

namespace tng {
namespace compat {
namespace {
const std::map<ge::DataType, std::string> kDataTypeToStringMap = {
    {ge::DataType::DT_UNDEFINED,      "DT_UNDEFINED"},          // Used to indicate a DataType field has not been set.
    {ge::DataType::DT_FLOAT,          "DT_FLOAT"},                    // float type
    {ge::DataType::DT_FLOAT16,        "DT_FLOAT16"},                // fp16 type
    {ge::DataType::DT_INT8,           "DT_INT8"},                      // int8 type
    {ge::DataType::DT_INT16,          "DT_INT16"},                    // int16 type
    {ge::DataType::DT_UINT16,         "DT_UINT16"},                  // uint16 type
    {ge::DataType::DT_UINT8,          "DT_UINT8"},                    // uint8 type
    {ge::DataType::DT_INT32,          "DT_INT32"},                    // uint32 type
    {ge::DataType::DT_INT64,          "DT_INT64"},                    // int64 type
    {ge::DataType::DT_UINT32,         "DT_UINT32"},                  // unsigned int32
    {ge::DataType::DT_UINT64,         "DT_UINT64"},                  // unsigned int64
    {ge::DataType::DT_BOOL,           "DT_BOOL"},                      // bool type
    {ge::DataType::DT_DOUBLE,         "DT_DOUBLE"},                  // double type
    {ge::DataType::DT_DUAL,           "DT_DUAL"},                      // dual output type
    {ge::DataType::DT_DUAL_SUB_INT8,  "DT_DUAL_SUB_INT8"},    // dual output int8 type
    {ge::DataType::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},  // dual output uint8 type
    {ge::DataType::DT_COMPLEX32,      "DT_COMPLEX32"},            // complex32 type
    {ge::DataType::DT_COMPLEX64,      "DT_COMPLEX64"},            // complex64 type
    {ge::DataType::DT_COMPLEX128,     "DT_COMPLEX128"},          // complex128 type
    {ge::DataType::DT_QINT8,          "DT_QINT8"},                    // qint8 type
    {ge::DataType::DT_QINT16,         "DT_QINT16"},                  // qint16 type
    {ge::DataType::DT_QINT32,         "DT_QINT32"},                  // qint32 type
    {ge::DataType::DT_QUINT8,         "DT_QUINT8"},                  // quint8 type
    {ge::DataType::DT_QUINT16,        "DT_QUINT16"},                // quint16 type
    {ge::DataType::DT_RESOURCE,       "DT_RESOURCE"},              // resource type
    {ge::DataType::DT_STRING_REF,     "DT_STRING_REF"},          // string ref type
    {ge::DataType::DT_STRING,         "DT_STRING"},                  // string type
    {ge::DataType::DT_VARIANT,        "DT_VARIANT"},                // dt_variant type
    {ge::DataType::DT_BF16,           "DT_BFLOAT16"},                  // dt_bfloat16 type
    {ge::DataType::DT_INT4,           "DT_INT4"},                      // dt_variant type
    {ge::DataType::DT_UINT1,          "DT_UINT1"},                    // dt_variant type
    {ge::DataType::DT_INT2,           "DT_INT2"},                      // dt_variant type
    {ge::DataType::DT_UINT2,          "DT_UINT2"}                     // dt_variant type
};

const std::map<ge::Format, std::string> kFormatToStringMap = {
    {ge::Format::FORMAT_NCHW,                            "NCHW"},
    {ge::Format::FORMAT_NHWC,                            "NHWC"},
    {ge::Format::FORMAT_ND,                              "ND"},
    {ge::Format::FORMAT_NC1HWC0,                         "NC1HWC0"},
    {ge::Format::FORMAT_FRACTAL_Z,                       "FRACTAL_Z"},
    {ge::Format::FORMAT_NC1C0HWPAD,                      "NC1C0HWPAD"},
    {ge::Format::FORMAT_NHWC1C0,                         "NHWC1C0"},
    {ge::Format::FORMAT_FSR_NCHW,                        "FSR_NCHW"},
    {ge::Format::FORMAT_FRACTAL_DECONV,                  "FRACTAL_DECONV"},
    {ge::Format::FORMAT_C1HWNC0,                         "C1HWNC0"},
    {ge::Format::FORMAT_FRACTAL_DECONV_TRANSPOSE,        "FRACTAL_DECONV_TRANSPOSE"},
    {ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,  "FRACTAL_DECONV_SP_STRIDE_TRANS"},
    {ge::Format::FORMAT_NC1HWC0_C04,                     "NC1HWC0_C04"},
    {ge::Format::FORMAT_FRACTAL_Z_C04,                   "FRACTAL_Z_C04"},
    {ge::Format::FORMAT_CHWN,                            "CHWN"},
    {ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS, "DECONV_SP_STRIDE8_TRANS"},
    {ge::Format::FORMAT_NC1KHKWHWC0,                     "NC1KHKWHWC0"},
    {ge::Format::FORMAT_BN_WEIGHT,                       "BN_WEIGHT"},
    {ge::Format::FORMAT_FILTER_HWCK,                     "FILTER_HWCK"},
    {ge::Format::FORMAT_HWCN,                            "HWCN"},
    {ge::Format::FORMAT_HASHTABLE_LOOKUP_LOOKUPS,        "LOOKUP_LOOKUPS"},
    {ge::Format::FORMAT_HASHTABLE_LOOKUP_KEYS,           "LOOKUP_KEYS"},
    {ge::Format::FORMAT_HASHTABLE_LOOKUP_VALUE,          "LOOKUP_VALUE"},
    {ge::Format::FORMAT_HASHTABLE_LOOKUP_OUTPUT,         "LOOKUP_OUTPUT"},
    {ge::Format::FORMAT_HASHTABLE_LOOKUP_HITS,           "LOOKUP_HITS"},
    {ge::Format::FORMAT_MD,                              "MD"},
    {ge::Format::FORMAT_NDHWC,                           "NDHWC"},
    {ge::Format::FORMAT_NCDHW,                           "NCDHW"},
    {ge::Format::FORMAT_DHWCN,                           "DHWCN"},
    {ge::Format::FORMAT_DHWNC,                           "DHWNC"},
    {ge::Format::FORMAT_NDC1HWC0,                        "NDC1HWC0"},
    {ge::Format::FORMAT_FRACTAL_Z_3D,                    "FRACTAL_Z_3D"},
    {ge::Format::FORMAT_FRACTAL_Z_3D_TRANSPOSE,          "FRACTAL_Z_3D_TRANSPOSE"},
    {ge::Format::FORMAT_C1HWNCoC0,                       "C1HWNCoC0"},
    {ge::Format::FORMAT_FRACTAL_NZ,                      "FRACTAL_NZ"},
    {ge::Format::FORMAT_CN,                              "CN"},
    {ge::Format::FORMAT_NC,                              "NC"},
    {ge::Format::FORMAT_FRACTAL_ZN_LSTM,                 "FRACTAL_ZN_LSTM"},
    {ge::Format::FORMAT_FRACTAL_Z_G,                     "FRACTAL_Z_G"},
    {ge::Format::FORMAT_ND_RNN_BIAS,                     "ND_RNN_BIAS"},
    {ge::Format::FORMAT_FRACTAL_ZN_RNN,                  "FRACTAL_ZN_RNN"},
    {ge::Format::FORMAT_NYUV,                            "NYUV"},
    {ge::Format::FORMAT_NYUV_A,                          "NYUV_A"},
    {ge::Format::FORMAT_NCL,                             "NCL"},
    {ge::Format::FORMAT_FRACTAL_Z_WINO,                  "FRACTAL_Z_WINO"},
    {ge::Format::FORMAT_RESERVED,                        "FORMAT_RESERVED"},
    {ge::Format::FORMAT_ALL,                             "ALL"},
    {ge::Format::FORMAT_NULL,                            "NULL"},
    {ge::Format::FORMAT_END,                             "END"},
    {ge::Format::FORMAT_MAX,                             "MAX"}
};
}  // namespace

using Name2Index = std::map<std::string, uint32_t>;
template <typename T>
Name2Index GetDescName2Index(const T &descs) {
  Name2Index name2index;
  for (auto &desc : descs) {
    name2index.emplace(desc.name(), name2index.size());
  }
  return name2index;
}

Status ParseGraphFromArray(const void *serialized_proto, size_t proto_size, ge::GraphPtr &graph) {
  TNG_ASSERT_NOTNULL(serialized_proto, "Given serialized proto is nullptr.");
  if (graph == nullptr) {
    graph = std::make_shared<ge::Graph>();
  }
  TNG_ASSERT(graph->LoadFromSerializedModelArray(serialized_proto, proto_size) == ge::GRAPH_SUCCESS);
  return Status::Success();
}

Status GeErrorStatus() {
  return Status::Error("%s", ge::GEGetErrorMsg().c_str());
}

Status DebugString(const ge::Shape &shape) {
  std::stringstream ss;
  auto dims = shape.GetDims();
  if (dims.empty()) {
    return Status::Error("[]");
  }
  ss << "[";
  size_t index = 0U;
  for (; index < (dims.size() - 1U); index++) {
    ss << dims[index] << ", ";
  }
  ss << dims[index] << "]";
  return Status::Error(ss.str().c_str());
}

Status DebugString(const gert::Shape &shape) {
  auto dims = GetDims(shape);
  if (dims.empty()) {
    return Status::Error("[]");
  }
  std::stringstream ss;
  ss << "[";
  size_t index = 0U;
  for (; index < (dims.size() - 1U); index++) {
    ss << dims[index] << ", ";
  }
  ss << dims[index] << "]";
  return Status::Error(ss.str().c_str());
}

Status DebugString(const ge::Tensor &tensor) {
  const auto &desc = tensor.GetTensorDesc();
  std::stringstream ss;
  ss << "ge::Tensor(shape=" << DebugString(desc.GetShape()).GetErrorMessage()
     << ", format=" << DebugString(desc.GetFormat()).GetErrorMessage()
     << ", dtype=" << DebugString(desc.GetDataType()).GetErrorMessage()
     << ", device=" << (desc.GetPlacement() == ge::Placement::kPlacementHost ? "CPU" : "NPU")
     << ", addr=" << static_cast<const void *>(tensor.GetData()) << ")";
  return Status::Error(ss.str().c_str());
}

Status DebugString(const gert::Tensor &tensor) {
  std::stringstream ss;
  ss << "ge::Tensor(storage shape=" << DebugString(tensor.GetStorageShape()).GetErrorMessage()
     << ", origin shape=" << DebugString(tensor.GetOriginShape()).GetErrorMessage()
     << ", storage format=" << DebugString(tensor.GetStorageFormat()).GetErrorMessage()
     << ", origin format=" << DebugString(tensor.GetOriginFormat()).GetErrorMessage()
     << ", dtype=" << DebugString(tensor.GetDataType()).GetErrorMessage()
     << ", device=" << (tensor.GetPlacement() == gert::TensorPlacement::kOnDeviceHbm ? "NPU" : "CPU")
     << ", addr=" << static_cast<const void *>(tensor.GetAddr()) << ")";
  return Status::Error(ss.str().c_str());
}

Status DebugString(const ge::DataType &dtype) {
  const auto iter = kDataTypeToStringMap.find(dtype);
  if (iter != kDataTypeToStringMap.end()) {
    return Status::Error(iter->second.c_str());
  }
  return Status::Error(("Unsupported dtype " + std::to_string(dtype)).c_str());
}

Status DebugString(const ge::Format &format) {
  const auto iter = kFormatToStringMap.find(format);
  if (iter != kFormatToStringMap.end()) {
    return Status::Error(iter->second.c_str());
  }
  return Status::Error(("Unsupported format " + std::to_string(format)).c_str());
}
}  // namespace compat
}  // namespace tng
