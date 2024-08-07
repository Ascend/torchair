/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_REGISTER_OP_COMPILE_INFO_BASE_H_
#define INC_EXTERNAL_REGISTER_OP_COMPILE_INFO_BASE_H_

#include <memory>

namespace optiling {
class CompileInfoBase;
using CompileInfoPtr = std::shared_ptr<CompileInfoBase>;

class CompileInfoBase {
public:
  CompileInfoBase() {}
  virtual ~CompileInfoBase() {}
};
}  // namespace optiling
#endif  // INC_REGISTER_OP_TILING_REGISTRY_H_
