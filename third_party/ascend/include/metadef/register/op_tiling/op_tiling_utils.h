/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef REGISTER_OP_TILING_OP_TILING_UTILS_H_
#define REGISTER_OP_TILING_OP_TILING_UTILS_H_

#include <vector>
#include <nlohmann/json.hpp>
#include "graph/op_desc.h"
#include "graph/debug/ge_log.h"

namespace optiling {
void ReplaceEmptyShapeOfTensorDesc(const ge::OpDescPtr &op_desc, std::vector<int32_t> &indexes);
void RecoveryEmptyShapeOfTensorDesc(const ge::OpDescPtr &op_desc, const std::vector<int32_t> &indexes);

#define OP_TILING_MAKE_SHARED(exec_expr0, exec_expr1) \
  do {                                                \
    try {                                             \
      exec_expr0;                                     \
    } catch (...) {                                   \
      GE_LOGE("Make shared failed");                  \
      exec_expr1;                                     \
    }                                                 \
  } while (0)

}  // namespace optiling
#endif  // REGISTER_OP_TILING_OP_TILING_UTILS_H_
