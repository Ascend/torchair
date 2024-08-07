/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
 
#ifndef INC_EXTERNAL_GE_FRAMEWORK_COMMON_TASKDOWN_COMMON_H_
#define INC_EXTERNAL_GE_FRAMEWORK_COMMON_TASKDOWN_COMMON_H_
 
namespace ge {
enum class ccKernelType : uint32_t {
  CCE_AI_CORE = 0, /* cce aicore */
  CCE_AI_CPU = 1,  /* cce aicpu */
  TE = 2,          /* te operator */
  CUSTOMIZED = 3,  /* customized operator */
  TE_AI_CORE = 4,  /* te aicore operator */
  TE_AI_CPU = 5,   /* te aicpu operator */
  AI_CPU = 6,      /* aicpu */
  CUST_AI_CPU = 7, /* custom aicpu */
  HOST_CPU = 8,    /* host cpu */
  DVPP = 9,        /* dvpp */
  AI_CPU_KFC = 10,  /* aicpu kfc */
  MIX_AICORE = 11,
  MIX_VECTOR_CORE = 12, /* vector core only */
  INVALID = 10000  /* unknown kernel type */
};
} // namespace ge
 
#endif  // INC_EXTERNAL_GE_FRAMEWORK_COMMON_TASKDOWN_COMMON_H_
