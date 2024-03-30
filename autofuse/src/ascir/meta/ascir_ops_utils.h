/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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

#ifndef AUTOFUSE_ASCIR_OPS_UTILS_H
#define AUTOFUSE_ASCIR_OPS_UTILS_H
#include "ascir.h"
namespace ascir {
namespace ops {
template <typename T>
bool IsOps(const ascir::NodeView &view) {
  return view->GetType() == T::Type;
}
}
namespace cg {
void AddToGraphFollowOp(const ge::Operator &op, ge::Operator &new_op);
}
}
#endif  // AUTOFUSE_ASCIR_OPS_UTILS_H
