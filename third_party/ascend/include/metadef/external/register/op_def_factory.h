/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef OP_DEF_FACTORY_H
#define OP_DEF_FACTORY_H

#include "register/op_def.h"

namespace ops {
using OpDefCreator = std::function<OpDef(const char *)>;
class OpDefFactory {
public:
  static int OpDefRegister(const char *name, OpDefCreator creator);
  static OpDef OpDefCreate(const char *name);
  static std::vector<ge::AscendString> &GetAllOp(void);
};
}  // namespace ops

#endif
