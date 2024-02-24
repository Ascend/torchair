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

#ifndef INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_
#define INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_
#include <initializer_list>
#include <string>
#include <map>
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/compiler_def.h"
#include "register/op_impl_kernel_registry.h"

namespace gert {
class OpImplRegisterV2Impl;
class OpImplRegisterV2 {
 public:
  explicit OpImplRegisterV2(const ge::char_t *op_type);
  OpImplRegisterV2(OpImplRegisterV2 &&register_data) noexcept;
  OpImplRegisterV2(const OpImplRegisterV2 &register_data);
  OpImplRegisterV2 &operator=(const OpImplRegisterV2 &) = delete;
  OpImplRegisterV2 &operator=(OpImplRegisterV2 &&) = delete;
  ~OpImplRegisterV2();

 public:
  OpImplRegisterV2 &InferShape(OpImplKernelRegistry::InferShapeKernelFunc infer_shape_func);
  OpImplRegisterV2 &InferShapeRange(OpImplKernelRegistry::InferShapeRangeKernelFunc infer_shape_range_func);
  OpImplRegisterV2 &InferDataType(OpImplKernelRegistry::InferDataTypeKernelFunc infer_datatype_func);
  OpImplRegisterV2 &Tiling(OpImplKernelRegistry::TilingKernelFunc tiling_func, size_t max_tiling_data_size = 2048);
  OpImplRegisterV2 &GenSimplifiedKey(OpImplKernelRegistry::GenSimplifiedKeyKernelFunc gen_simplifiedkey_func);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, int64_t private_attr_val);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, const std::vector<int64_t> &private_attr_val);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, const ge::char_t *private_attr_val);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, ge::float32_t private_attr_val);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, bool private_attr_val);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, const std::vector<ge::float32_t> &private_attr_val);
  template<typename T>
  OpImplRegisterV2 &TilingParse(OpImplKernelRegistry::KernelFunc const tiling_parse_func) {
    return TilingParse(tiling_parse_func, CreateCompileInfo<T>, DeleteCompileInfo<T>);
  }
  template<typename T>
  OpImplRegisterV2 &TilingParse(OpImplKernelRegistry::TilingParseFunc const tiling_parse_func) {
    return TilingParse(reinterpret_cast<OpImplKernelRegistry::KernelFunc>(tiling_parse_func), CreateCompileInfo<T>,
                       DeleteCompileInfo<T>);
  }
  OpImplRegisterV2 &InputsDataDependency(std::initializer_list<int32_t> inputs);
  OpImplRegisterV2 &OpExecuteFunc(OpImplKernelRegistry::OpExecuteFunc op_execute_func);
  OpImplRegisterV2 &HostInputs(std::initializer_list<int32_t> inputs);
  OpImplRegisterV2 &TilingInputsDataDependency(std::initializer_list<int32_t> inputs);
 private:
  OpImplRegisterV2 &TilingParse(OpImplKernelRegistry::KernelFunc tiling_parse_func,
                                OpImplKernelRegistry::CompileInfoCreatorFunc creator_func,
                                OpImplKernelRegistry::CompileInfoDeleterFunc deleter_func);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, ge::AnyValue private_attr_av);

  template<typename T, typename std::enable_if<(!std::is_array<T>::value), int32_t>::type = 0>
  static void *CreateCompileInfo() {
    return new T();
  }
  template<typename T>
  static void DeleteCompileInfo(void *const obj) {
    delete reinterpret_cast<T *>(obj);
  }
 private:
  std::unique_ptr<OpImplRegisterV2Impl> impl_;
};
}  // namespace gert

#define IMPL_OP_COUNTER(op_type, name, counter) \
  static gert::OpImplRegisterV2 VAR_UNUSED name##counter = gert::OpImplRegisterV2(#op_type)
#define IMPL_OP_COUNTER_NUMBER(op_type, name, counter) IMPL_OP_COUNTER(op_type, name, counter)
#define IMPL_OP(op_type) IMPL_OP_COUNTER_NUMBER(op_type, op_impl_register_##op_type, __COUNTER__)
#define IMPL_OP_DEFAULT() IMPL_OP(DefaultImpl)

#endif  // INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_
