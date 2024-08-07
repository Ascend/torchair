/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_REGISTER_OP_CT_IMPL_REGISTRY_H_
#define INC_EXTERNAL_REGISTER_OP_CT_IMPL_REGISTRY_H_
#include <string>
#include <map>
#include "graph/types.h"
#include "graph/compiler_def.h"
#include "register/op_ct_impl_kernel_registry.h"

namespace gert {
class OpCtImplRegistry : public OpCtImplKernelRegistry {
 public:
  static OpCtImplRegistry &GetInstance();
  OpCtImplFunctions &CreateOrGetOpImpl(const ge::char_t *op_type);
  const OpCtImplFunctions *GetOpImpl(const ge::char_t *op_type) const;
  const std::map<OpType, OpCtImplFunctions> &GetAllTypesToImpl() const;
  std::map<OpType, OpCtImplFunctions> &GetAllTypesToImpl();

 private:
  std::map<OpType, OpCtImplFunctions> types_to_impl_;
  uint8_t reserved_[40] = {0U};  // Reserved field, 32+8, do not directly use when only 8-byte left
};

class OpCtImplRegisterV2Impl {
 public:
  OpCtImplRegistry::OpType op_type;
  OpCtImplRegistry::OpCtImplFunctions functions;
};

class OpCtImplRegisterV2 {
 public:
  explicit OpCtImplRegisterV2(const ge::char_t *op_type);
  OpCtImplRegisterV2(OpCtImplRegisterV2 &&register_data) noexcept;
  OpCtImplRegisterV2(const OpCtImplRegisterV2 &register_data);
  OpCtImplRegisterV2 &operator=(const OpCtImplRegisterV2 &) = delete;
  OpCtImplRegisterV2 &operator=(OpCtImplRegisterV2 &&) = delete;
  ~OpCtImplRegisterV2();

 public:
  OpCtImplRegisterV2 &CalcOpParam(OpCtImplKernelRegistry::OpCalcParamKernelFunc calc_op_param_func);
  OpCtImplRegisterV2 &GenerateTask(OpCtImplKernelRegistry::OpGenTaskKernelFunc gen_task_func);
 private:
  std::unique_ptr<OpCtImplRegisterV2Impl> impl_;
};
}  // namespace gert

#define IMPL_OP_CT_COUNTER(op_type, name, counter) \
  static gert::OpCtImplRegisterV2 VAR_UNUSED name##counter = gert::OpCtImplRegisterV2(#op_type)
#define IMPL_OP_CT_COUNTER_NUMBER(op_type, name, counter) IMPL_OP_CT_COUNTER(op_type, name, counter)
#define IMPL_OP_CT(op_type) IMPL_OP_CT_COUNTER_NUMBER(op_type, op_impl_register_##op_type, __COUNTER__)

#endif  // INC_EXTERNAL_REGISTER_OP_CT_IMPL_REGISTRY_H_
