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

#ifndef OP_DEF_REGISTRY_H
#define OP_DEF_REGISTRY_H

#include "register/op_def.h"
#include "register/op_def_factory.h"

#if defined(OP_PROTO_LIB)

#define OP_ADD(opType, ...)                                                                                            \
  static int g_##opType##_added = [](const char *name) {                                                               \
    opType op(#opType);                                                                                                \
    op.OpProtoPost(#opType);                                                                                           \
    return 0;                                                                                                          \
  }(#opType)

#elif defined(OP_TILING_LIB)

#define OP_ADD(opType, compInfo)                                                                                       \
  static int g_##opType##_added = [](const char *name) {                                                               \
    opType op(#opType);                                                                                                \
    op.AICore().OpTilingPost<compInfo>(#opType);                                                                       \
    op.AICore().OpCheckPost(#opType);                                                                                  \
    return 0;                                                                                                          \
  }(#opType)

#else

#define OP_ADD(opType, ...)                                                                                            \
  static int g_##opType##_added =                                                                                      \
      ops::OpDefFactory::OpDefRegister(#opType, [](const char *name) { return opType(#opType); })

#endif
#endif
