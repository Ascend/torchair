/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_REGISTER_OP_CT_IMPL_KERNEL_REGISTRY_H_
#define INC_EXTERNAL_REGISTER_OP_CT_IMPL_KERNEL_REGISTRY_H_
#include "exe_graph/runtime/base_type.h"
#include "graph/ascend_string.h"
#include "exe_graph/runtime/exe_res_generation_context.h"

#define OP_CT_IMPL_MAIM_VERSION 1

namespace gert {
struct OpCtImplKernelRegistry {
  using OpType = ge::AscendString;
  using OpCalcParamKernelFunc = UINT32 (*)(ExeResGenerationContext *context);
  using OpGenTaskKernelFunc = UINT32 (*)(const ExeResGenerationContext *context,
                                         std::vector<std::vector<uint8_t>> &tasks);
  struct OpCtImplFunctions {
    uint32_t st_size = sizeof(OpCtImplFunctions);
    uint32_t version = OP_CT_IMPL_MAIM_VERSION;
    OpCalcParamKernelFunc calc_op_param = nullptr;
    OpGenTaskKernelFunc gen_task = nullptr;;
  };
};
}  // namespace gert
#endif
