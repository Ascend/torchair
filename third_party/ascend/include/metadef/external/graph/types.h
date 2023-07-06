/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INC_EXTERNAL_GRAPH_TYPES_H_
#define INC_EXTERNAL_GRAPH_TYPES_H_

#include <atomic>
#include <memory>
#include <vector>

namespace ge {
using char_t = char;
using float32_t = float;
using float64_t = double;
using vector_bit_t = std::vector<bool>;

static const int64_t UNKNOWN_DIM = -1;
static const int64_t UNKNOWN_DIM_NUM = -2;
static const std::vector<int64_t> UNKNOWN_SHAPE = {-1};
static const std::vector<int64_t> UNKNOWN_RANK = {-2};
static const std::vector<int64_t> DUMMY_SHAPE = {-3};
// When data type unit is bit, this offset need to be added.
static constexpr int32_t kDataTypeSizeBitOffset = 1000;
static constexpr uint32_t kBitNumOfOneByte = 8U;
static constexpr uint32_t kBitThreeBytes = 24U;

#if(defined(HOST_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif
#if(defined(DEV_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_DEV_VISIBILITY
#endif

enum DataType {
  DT_FLOAT = 0,            // float type
  DT_FLOAT16 = 1,          // fp16 type
  DT_INT8 = 2,             // int8 type
  DT_INT16 = 6,            // int16 type
  DT_UINT16 = 7,           // uint16 type
  DT_UINT8 = 4,            // uint8 type
  DT_INT32 = 3,            //
  DT_INT64 = 9,            // int64 type
  DT_UINT32 = 8,           // unsigned int32
  DT_UINT64 = 10,          // unsigned int64
  DT_BOOL = 12,            // bool type
  DT_DOUBLE = 11,          // double type
  DT_STRING = 13,          // string type
  DT_DUAL_SUB_INT8 = 14,   // dual output int8 type
  DT_DUAL_SUB_UINT8 = 15,  // dual output uint8 type
  DT_COMPLEX64 = 16,       // complex64 type
  DT_COMPLEX128 = 17,      // complex128 type
  DT_QINT8 = 18,           // qint8 type
  DT_QINT16 = 19,          // qint16 type
  DT_QINT32 = 20,          // qint32 type
  DT_QUINT8 = 21,          // quint8 type
  DT_QUINT16 = 22,         // quint16 type
  DT_RESOURCE = 23,        // resource type
  DT_STRING_REF = 24,      // string ref type
  DT_DUAL = 25,            // dual output type
  DT_VARIANT = 26,         // dt_variant type
  DT_BF16 = 27,            // bf16 type
  DT_UNDEFINED = 28,       // Used to indicate a DataType field has not been set.
  DT_INT4 = 29,            // int4 type
  DT_UINT1 = 30,           // uint1 type
  DT_INT2 = 31,            // int2 type
  DT_UINT2 = 32,           // uint2 type
  DT_MAX                   // Mark the boundaries of data types
};

inline int GetSizeByDataType(DataType data_type) {
  static int data_type_size[DT_MAX] = {
      4,   // DT_FLOAT = 0,               float type
      2,   // DT_FLOAT16 = 1,             fp16 type
      1,   // DT_INT8 = 2,                int8 type
      4,   // DT_INT32 = 3,
      1,   // DT_UINT8 = 4,               uint8 type
      -1,
      2,   // DT_INT16 = 6,               int16 type
      2,   // DT_UINT16 = 7,              uint16 type
      4,   // DT_UINT32 = 8,              unsigned int32
      8,   // DT_INT64 = 9,               int64 type
      8,   // DT_UINT64 = 10,             unsigned int64
      8,   // DT_DOUBLE = 11,             double type
      1,   // DT_BOOL = 12,               bool type
      -1,  // DT_STRING = 13,             string type
      1,   // DT_DUAL_SUB_INT8 = 14,      dual output int8 type
      1,   // DT_DUAL_SUB_UINT8 = 15,     dual output uint8 type
      8,   // DT_COMPLEX64 = 16,          complex64 type
      16,  // DT_COMPLEX128 = 17,         complex128 type
      1,   // DT_QINT8 = 18,              qint8 type
      2,   // DT_QINT16 = 19,             qint16 type
      4,   // DT_QINT32 = 20,             qint32 type
      1,   // DT_QUINT8 = 21,             quint8 type
      2,   // DT_QUINT16 = 22,            quint16 type
      8,   // DT_RESOURCE = 23,           resource type
      -1,  // DT_STRING_REF = 24,         string ref type
      5,   // DT_DUAL = 25,               dual output type (float + int8)
      8,   // DT_VARIANT                  variant type
      2,   // DT_BF16 = 27,               bf16 type
      -1,  // DT_UNDEFINED = 28           Used to indicate a DataType field has not been set.
      kDataTypeSizeBitOffset + 4,    // DT_INT4 = 29,             int4 type
      kDataTypeSizeBitOffset + 1,    // DT_UINT1 = 30,            uint1 type
      kDataTypeSizeBitOffset + 2,    // DT_INT2 = 31,             int2 type
      kDataTypeSizeBitOffset + 2,    // DT_UINT2 = 32,            uint2 type
           // DT_MAX
  };
  if (data_type >= DT_MAX) {
    return -1;
  }
  return data_type_size[data_type];
}

/// @brief Calculates the length in bytes based on the DataType and the number of elements.
/// @param element_count
/// @param data_type
/// @return
int64_t GetSizeInBytes(int64_t element_count, DataType data_type);

enum Format {
  FORMAT_NCHW = 0,   // NCHW
  FORMAT_NHWC,       // NHWC
  FORMAT_ND,         // Nd Tensor
  FORMAT_NC1HWC0,    // NC1HWC0
  FORMAT_FRACTAL_Z,  // FRACTAL_Z
  FORMAT_NC1C0HWPAD = 5,
  FORMAT_NHWC1C0,
  FORMAT_FSR_NCHW,
  FORMAT_FRACTAL_DECONV,
  FORMAT_C1HWNC0,
  FORMAT_FRACTAL_DECONV_TRANSPOSE = 10,
  FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
  FORMAT_NC1HWC0_C04,    // NC1HWC0, C0 is 4
  FORMAT_FRACTAL_Z_C04,  // FRACZ, C0 is 4
  FORMAT_CHWN,
  FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15,
  FORMAT_HWCN,
  FORMAT_NC1KHKWHWC0,  // KH,KW kernel h& kernel w maxpooling max output format
  FORMAT_BN_WEIGHT,
  FORMAT_FILTER_HWCK,  // filter input tensor format
  FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20,
  FORMAT_HASHTABLE_LOOKUP_KEYS,
  FORMAT_HASHTABLE_LOOKUP_VALUE,
  FORMAT_HASHTABLE_LOOKUP_OUTPUT,
  FORMAT_HASHTABLE_LOOKUP_HITS,
  FORMAT_C1HWNCoC0 = 25,
  FORMAT_MD,
  FORMAT_NDHWC,
  FORMAT_FRACTAL_ZZ,
  FORMAT_FRACTAL_NZ,
  FORMAT_NCDHW = 30,
  FORMAT_DHWCN,  // 3D filter input tensor format
  FORMAT_NDC1HWC0,
  FORMAT_FRACTAL_Z_3D,
  FORMAT_CN,
  FORMAT_NC = 35,
  FORMAT_DHWNC,
  FORMAT_FRACTAL_Z_3D_TRANSPOSE, // 3D filter(transpose) input tensor format
  FORMAT_FRACTAL_ZN_LSTM,
  FORMAT_FRACTAL_Z_G,
  FORMAT_RESERVED = 40,
  FORMAT_ALL,
  FORMAT_NULL,
  FORMAT_ND_RNN_BIAS,
  FORMAT_FRACTAL_ZN_RNN,
  FORMAT_NYUV = 45,
  FORMAT_NYUV_A,
  FORMAT_NCL,
  // Add new formats definition here
  FORMAT_END,
  // FORMAT_MAX defines the max value of Format.
  // Any Format should not exceed the value of FORMAT_MAX.
  // ** Attention ** : FORMAT_MAX stands for the SPEC of enum Format and almost SHOULD NOT be used in code.
  //                   If you want to judge the range of Format, you can use FORMAT_END.
  FORMAT_MAX = 0xff
};

/// Get format from primary and sub-format,
/// in bits field:
/// ---------------------------------------------
/// |   4bits  |   4bits   |   2 bytes  | 1 byte |
/// |----------|-----------|------------|--------|
/// | reserved | c0_format | sub-format | format |
/// ---------------------------------------------
/// @param primary_format
/// @param sub_format
/// @param c0_format
/// @return
inline int32_t GetFormatFromSub(int32_t primary_format, int32_t sub_format) {
  return static_cast<int32_t>((static_cast<uint32_t>(primary_format) & 0xffU) |
                              ((static_cast<uint32_t>(sub_format) & 0xffffU) << kBitNumOfOneByte));
}

inline int32_t GetFormatFromC0(int32_t format, int32_t c0_format) {
  return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffffffU) |
                              ((static_cast<uint32_t>(c0_format) & 0xfU) << kBitThreeBytes));
}

inline int32_t GetFormatFromSubAndC0(int32_t primary_format, int32_t sub_format, int32_t c0_format) {
  return static_cast<int32_t>((static_cast<uint32_t>(primary_format) & 0xffU) |
                              ((static_cast<uint32_t>(sub_format) & 0xffffU) << kBitNumOfOneByte) |
                              ((static_cast<uint32_t>(c0_format) & 0xfU) << kBitThreeBytes));
}

inline int32_t GetPrimaryFormat(int32_t format) {
  return static_cast<int32_t>(static_cast<uint32_t>(format) & 0xffU);
}

inline int32_t GetSubFormat(int32_t format) {
  return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffff00U) >> kBitNumOfOneByte);
}

inline bool HasSubFormat(int32_t format) {
  return GetSubFormat(format) > 0;
}

inline int64_t GetC0Value(int32_t format) {
  return static_cast<int64_t>(1 <<
      (static_cast<int32_t>((static_cast<uint32_t>(format) & 0xf000000U) >> kBitThreeBytes) - 1));
}

inline bool HasC0Format(int32_t format) {
  return ((static_cast<uint32_t>(format) & 0xf000000U) >> kBitThreeBytes) > 0;
}

// for unknown shape op type
enum UnknowShapeOpType {
  DEPEND_IN_SHAPE    = 1,  // op out shape get by input shape
  DEPEND_CONST_VALUE = 2,  // op out shape get by const op value
  DEPEND_SHAPE_RANGE = 3,  // op out shape get by range
  DEPEND_COMPUTE     = 4   // op out shape get by totally computing
};

struct TensorDescInfo {
  Format format_ = FORMAT_RESERVED;  // tbe op register support format
  DataType dataType_ = DT_UNDEFINED; // tbe op register support datatype
};

enum DeviceType {
  NPU = 0,
  CPU = 1,
};

enum Placement {
  kPlacementHost = 0,     // host data addr
  kPlacementDevice = 1,   // device data addr
};

///
/// @brief Get a format name from enum
/// @param format
/// @return
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
const char_t *GetFormatName(Format format);

class TensorTypeImpl;
struct TensorType {
  explicit TensorType(DataType dt);

  TensorType(const std::initializer_list<DataType> &initial_types);

  static TensorType ALL() {
    return TensorType{DT_BOOL,   DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                      DT_INT32,  DT_INT64,      DT_INT8,      DT_QINT16, DT_QINT32, DT_QINT8,   DT_QUINT16,
                      DT_QUINT8, DT_RESOURCE,   DT_STRING,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                      DT_BF16};
  }

  static TensorType QuantifiedType() { return TensorType{DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8}; }

  static TensorType OrdinaryType() {
    return TensorType{DT_BOOL,  DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                      DT_INT32, DT_INT64,      DT_INT8,      DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                      DT_BF16};
  }

  static TensorType BasicType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                      DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                      DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                      DT_BF16};
  }

  static TensorType NumberType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,  DT_INT32,  DT_INT64,
                      DT_INT8,       DT_QINT32,    DT_QINT8,  DT_QUINT8, DT_UINT16,  DT_UINT32, DT_UINT64, DT_UINT8,
                      DT_BF16};
  }

  static TensorType RealNumberType() {
    return TensorType{DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,  DT_INT32, DT_INT64,
                      DT_INT8,   DT_UINT16, DT_UINT32,  DT_UINT64, DT_UINT8, DT_BF16};
  }

  static TensorType ComplexDataType() { return TensorType{DT_COMPLEX128, DT_COMPLEX64}; }

  static TensorType IntegerDataType() {
    return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
  }

  static TensorType SignedDataType() { return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8}; }

  static TensorType UnsignedDataType() { return TensorType{DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}; }

  static TensorType FloatingDataType() { return TensorType{DT_DOUBLE, DT_FLOAT, DT_FLOAT16}; }

  static TensorType IndexNumberType() { return TensorType{DT_INT32, DT_INT64}; }

  static TensorType UnaryDataType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_BF16};
  }

  static TensorType FLOAT() { return TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}; }

  std::shared_ptr<TensorTypeImpl> tensor_type_impl_;
};

struct ListTensorType {
  explicit ListTensorType(const TensorType &type) : tensor_type(type) {};
  TensorType tensor_type;
};
}  // namespace ge

namespace domi {
enum class ImplyType : unsigned int {
  BUILDIN = 0,  // Built in operator, normally executed by OME
  TVM,          // Compile to TVM bin file for execution
  CUSTOM,       // User defined calculation logic, executed by CPU
  AI_CPU,       // AICPU
  CCE,          // Cce
  GELOCAL,      // GE local, do node need execute by device
  HCCL,         // Hccl
  INVALID = 0xFFFFFFFF,
};
using char_t = ge::char_t;
using float32_t = ge::float32_t;
using float64_t = ge::float64_t;
}  // namespace domi

#endif  // INC_EXTERNAL_GRAPH_TYPES_H_
