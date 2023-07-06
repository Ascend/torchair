/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
