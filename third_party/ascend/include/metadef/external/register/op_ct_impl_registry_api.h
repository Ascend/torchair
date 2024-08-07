/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_REGISTER_OP_CT_IMPL_REGISTRY_API_H_
#define INC_EXTERNAL_REGISTER_OP_CT_IMPL_REGISTRY_API_H_

#include <cstdlib>
#include "op_ct_impl_kernel_registry.h"

struct TypesToCtImpl {
  const char *op_type;
  gert::OpCtImplKernelRegistry::OpCtImplFunctions funcs;
};

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GNUC__
#define METADEF_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define METADEF_FUNC_VISIBILITY
#endif

METADEF_FUNC_VISIBILITY size_t GetRegisteredOpCtNum(void);
METADEF_FUNC_VISIBILITY int32_t GetOpCtImplFunctions(TypesToCtImpl *impl, size_t impl_num);

#ifdef __cplusplus
}
#endif

#endif  // INC_EXTERNAL_REGISTER_OP_CT_IMPL_REGISTRY_API_H_
