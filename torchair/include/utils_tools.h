/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TORCH_AIR_TORCH_AIR_UTILS_TOOLS_H_
#define TORCH_AIR_TORCH_AIR_UTILS_TOOLS_H_

#include "executor.h"
#include "export.h"
#include "tng_status.h"
#include "torch/torch.h"

namespace tng {
class NpuOpUtilsTools {
 public:
  static bool CheckAclnnAvaliable(const std::string &aclnn_name);
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_UTILS_TOOLS_H_
