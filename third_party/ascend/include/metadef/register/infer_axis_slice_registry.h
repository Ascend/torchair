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

#ifndef INC_REGISTER_INFER_AXIS_SLICE_REGISTRY_H_
#define INC_REGISTER_INFER_AXIS_SLICE_REGISTRY_H_

#include "external/graph/ge_error_codes.h"
#include "external/graph/operator.h"
#include "external/graph/types.h"
#include "graph/axis_type_info.h"

namespace ge {
// cut tensor : axis index : slice range
using DataSliceInfo = std::vector<std::vector<std::vector<int64_t>>>;
using InferAxisTypeInfoFunc = std::function<graphStatus(Operator &, std::vector<AxisTypeInfo> &)>;
using InferAxisSliceFunc = std::function<graphStatus(Operator &, const AxisTypeInfo &, const DataSliceInfo &,
                                                     DataSliceInfo &)>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferAxisTypeInfoFuncRegister {
 public:
  InferAxisTypeInfoFuncRegister(const std::string &operator_type,
                                const InferAxisTypeInfoFunc &infer_axis_type_info_func);
  InferAxisTypeInfoFuncRegister(const char_t *const operator_type,
                                const InferAxisTypeInfoFunc &infer_axis_type_info_func);
  ~InferAxisTypeInfoFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferAxisSliceFuncRegister {
 public:
  InferAxisSliceFuncRegister(const std::string &operator_type, const InferAxisSliceFunc &infer_axis_slice_func);
  InferAxisSliceFuncRegister(const char_t *const operator_type, const InferAxisSliceFunc &infer_axis_slice_func);
  ~InferAxisSliceFuncRegister() = default;
};
}

#define PASTE(g_register, y) g_register##y

// infer axis type info func register
#define IMPLEMT_COMMON_INFER_AXIS_TYPE_INFO(func_name)                                                \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static ge::graphStatus (func_name)(ge::Operator &op, \
      std::vector<ge::AxisTypeInfo> &axis_type)

#define IMPLEMT_INFER_AXIS_TYPE_INFO(op_name, func_name)                                           \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static ge::graphStatus (func_name)((op_name) &op, \
      std::vector<AxisTypeInfo> &axis_type)

#define INFER_AXIS_TYPE_INFO_FUNC(op_name, x) [](ge::Operator &v, std::vector<ge::AxisTypeInfo> &axis_type) \
  { return (x)(v, axis_type); }

#define INFER_AXIS_TYPE_INFO_REG_IMPL(op_name, x, n) \
  static const ge::InferAxisTypeInfoFuncRegister PASTE(ids_register, n)(#op_name, (x))

#define INFER_AXIS_TYPE_INFO_REG(op_name, x) \
  INFER_AXIS_TYPE_INFO_REG_IMPL(op_name, INFER_AXIS_TYPE_INFO_FUNC(op_name, x), __COUNTER__)

// infer axis slice func register
#define IMPLEMT_COMMON_INFER_AXIS_SLICE(func_name)                                                              \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static ge::graphStatus (func_name)(ge::Operator &op,           \
      const ge::AxisTypeInfo &axis_info, const ge::DataSliceInfo &output_param, ge::DataSliceInfo &input_param)

#define IMPLEMT_INFER_AXIS_SLICE(op_name, func_name)                                                            \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static ge::graphStatus (func_name)((op_name) &op,              \
      const ge::AxisTypeInfo &axis_info, const ge::DataSliceInfo &output_param, ge::DataSliceInfo &input_param)

#define INFER_AXIS_SLICE_FUNC(op_name, x) [](ge::Operator &v, const ge::AxisTypeInfo &axis_info,                    \
                                              const ge::DataSliceInfo &output_param, ge::DataSliceInfo &input_param) \
  { return (x)(v, axis_info, output_param, input_param); }

#define INFER_AXIS_SLICE_FUNC_REG_IMPL(op_name, x, n) \
  static const ge::InferAxisSliceFuncRegister PASTE(ids_register, n)(#op_name, (x))

#define INFER_AXIS_SLICE_FUNC_REG(op_name, x) \
  INFER_AXIS_SLICE_FUNC_REG_IMPL(op_name, INFER_AXIS_SLICE_FUNC(op_name, x), __COUNTER__)

#endif  // INC_REGISTER_INFER_AXIS_SLICE_REGISTRY_H_
