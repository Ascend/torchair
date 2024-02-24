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
    gert::OpImplRegisterV2 impl(#opType);                                                                              \
    impl.InferShape(op.GetInferShape())                                                                                \
      .InferShapeRange(op.GetInferShapeRange())                                                                        \
      .InferDataType(op.GetInferDataType());                                                                           \
    gert::OpImplRegisterV2 implReg(impl);                                                                              \
    return 0;                                                                                                          \
  }(#opType)

#elif defined(OP_TILING_LIB)

#define OP_ADD(opType, ...)                                                                                            \
  struct OpAddCompilerInfoPlaceholder##opType {};                                                                      \
  static ge::graphStatus TilingPrepare##opType(gert::TilingParseContext *context) { return ge::GRAPH_SUCCESS; }        \
  static int g_##opType##_added = [](const char *name) {                                                               \
    opType op(#opType);                                                                                                \
    gert::OpImplRegisterV2 impl(#opType);                                                                              \
    impl.Tiling(op.AICore().GetTiling());                                                                              \
    impl.TilingParse<OpAddCompilerInfoPlaceholder##opType>(TilingPrepare##opType);                                     \
    optiling::OpCheckFuncHelper(FUNC_CHECK_SUPPORTED, #opType, op.AICore().GetCheckSupport());                         \
    optiling::OpCheckFuncHelper(FUNC_OP_SELECT_FORMAT, #opType, op.AICore().GetOpSelectFormat());                      \
    optiling::OpCheckFuncHelper(FUNC_GET_OP_SUPPORT_INFO, #opType, op.AICore().GetOpSupportInfo());                    \
    optiling::OpCheckFuncHelper(FUNC_GET_SPECIFIC_INFO, #opType, op.AICore().GetOpSpecInfo());                         \
    optiling::OpCheckFuncHelper(#opType, op.AICore().GetParamGeneralize());                                            \
    gert::OpImplRegisterV2 implReg(impl);                                                                              \
    return 0;                                                                                                          \
  }(#opType)

#else

#define OP_ADD(opType, ...)                                                                                            \
  static int g_##opType##_added =                                                                                      \
      ops::OpDefFactory::OpDefRegister(#opType, [](const char *name) { return opType(#opType); })

#endif
#endif
