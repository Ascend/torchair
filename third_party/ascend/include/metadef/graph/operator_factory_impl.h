/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#ifndef INC_GRAPH_OPERATOR_FACTORY_IMPL_H_
#define INC_GRAPH_OPERATOR_FACTORY_IMPL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "graph/operator_factory.h"
#include "register/infer_data_slice_registry.h"
#include "register/infer_axis_slice_registry.h"
#include "register/op_impl_kernel_registry.h"
#include "graph/op_desc.h"

namespace ge {
using InferShapeV2Func = uint32_t (*)(const ge::Operator &op, const OpDescPtr &);
using InferDataTypeFunc = uint32_t (*)(const OpDescPtr &);
using InferShapeRangeFunc = uint32_t (*)(const ge::Operator &op, const OpDescPtr &);
struct InferValueRangePara {
 public:
  InferValueRangePara() = default;
  InferValueRangePara(const WHEN_CALL call, const bool cpu_kernel, const InferValueRangeFunc func) {
    is_initialized = true;
    use_cpu_kernel = cpu_kernel;
    when_call = call;
    infer_value_func = func;
  }
  friend class OpDescImpl;
  friend class InferValueRangePass;
  friend class OpDescUtilsEx;
  ~InferValueRangePara() = default;
private:
  bool is_initialized = false;
  bool use_cpu_kernel = false;
  WHEN_CALL when_call = INPUT_IS_DYNAMIC;
  InferValueRangeFunc infer_value_func = nullptr;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorFactoryImpl {
 public:
  static Operator CreateOperator(const std::string &operator_name, const std::string &operator_type);

  static graphStatus GetOpsTypeList(std::vector<std::string> &all_ops);

  static bool IsExistOp(const std::string &operator_type);

  static InferShapeFunc GetInferShapeFunc(const std::string &operator_type);

  static InferShapeV2Func GetInferShapeV2Func();

  static InferDataTypeFunc GetInferDataTypeFunc();

  static InferShapeRangeFunc GetInferShapeRangeFunc();

  static InferFormatFunc GetInferFormatFunc(const std::string &operator_type);

  static InferValueRangePara GetInferValueRangePara(const std::string &operator_type);

  static VerifyFunc GetVerifyFunc(const std::string &operator_type);

  static InferDataSliceFunc GetInferDataSliceFunc(const std::string &operator_type);

  static InferAxisSliceFunc GetInferAxisSliceFunc(const std::string &operator_type);

  static InferAxisTypeInfoFunc GetInferAxisTypeInfoFunc(const std::string &operator_type);

  static void SetRegisterOverridable(const bool &is_overridable);

  static graphStatus RegisterOperatorCreator(const std::string &operator_type, OpCreator const &op_creator);

  static graphStatus RegisterOperatorCreator(const std::string &operator_type, OpCreatorV2 const &op_creator);

  static graphStatus RegisterInferShapeFunc(const std::string &operator_type, InferShapeFunc const infer_shape_func);

  static void RegisterInferShapeV2Func(InferShapeV2Func const infer_shape_func);

  static void RegisterInferDataTypeFunc(InferDataTypeFunc const infer_data_type_func);

  static void RegisterInferShapeRangeFunc(InferShapeRangeFunc const infer_shape_range_func);

  static graphStatus RegisterInferFormatFunc(const std::string &operator_type, InferFormatFunc const infer_format_func);

  static graphStatus RegisterVerifyFunc(const std::string &operator_type, VerifyFunc const verify_func);

  static graphStatus RegisterInferDataSliceFunc(const std::string &operator_type,
                                                InferDataSliceFunc const infer_data_slice_func);

  static graphStatus RegisterInferValueRangeFunc(const std::string &operator_type);

  static graphStatus RegisterInferValueRangeFunc(const std::string &operator_type,
                                                 const WHEN_CALL when_call,
                                                 const bool use_cpu_kernel,
                                                 const InferValueRangeFunc &infer_value_range_func);

  static graphStatus RegisterInferAxisSliceFunc(const std::string &operator_type,
                                                const InferAxisSliceFunc &infer_axis_slice_func);

  static graphStatus RegisterInferAxisTypeInfoFunc(const std::string &operator_type,
                                                   const InferAxisTypeInfoFunc &infer_axis_type_info_func);

  static std::shared_ptr<std::map<std::string, OpCreator>> operator_creators_;
  static std::shared_ptr<std::map<std::string, OpCreatorV2>> operator_creators_v2_;
  static std::shared_ptr<std::map<std::string, InferShapeFunc>> operator_infershape_funcs_;
  static std::shared_ptr<std::map<std::string, InferFormatFunc>> operator_inferformat_funcs_;
  static std::shared_ptr<std::map<std::string, VerifyFunc>> operator_verify_funcs_;
  static std::shared_ptr<std::map<std::string, InferDataSliceFunc>> operator_infer_data_slice_funcs_;
  static std::shared_ptr<std::map<std::string, InferValueRangePara>> operator_infer_value_range_paras_;
  static std::shared_ptr<std::map<std::string, InferAxisSliceFunc>> operator_infer_axis_slice_funcs_;
  static std::shared_ptr<std::map<std::string, InferAxisTypeInfoFunc>> operator_infer_axis_type_info_funcs_;
  static InferShapeV2Func operator_infer_shape_v2_func_;
  static InferDataTypeFunc operator_infer_datatype_func_;
  static InferShapeRangeFunc operator_infer_shape_range_func_;
};
}  // namespace ge

#endif  // INC_GRAPH_OPERATOR_FACTORY_IMPL_H_
