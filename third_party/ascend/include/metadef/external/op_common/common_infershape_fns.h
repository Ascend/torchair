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

/*!
 * \file common_infershape_fns.h
 * \brief
 */

#ifndef EXTERNAL_OP_COMMON_INFERSHAPE_FNS_H_
#define EXTERNAL_OP_COMMON_INFERSHAPE_FNS_H_

#include "external/exe_graph/runtime/shape.h"
#include "external/exe_graph/runtime/infer_shape_context.h"

namespace opcommon {
ge::graphStatus InferShape4BroadcastOp(gert::InferShapeContext* context);
ge::graphStatus InferShape4ReduceOp(gert::InferShapeContext* context);
ge::graphStatus InferShape4ElewiseOp(gert::InferShapeContext* context);
} // namespace opcommon

#endif // EXTERNAL_OP_COMMON_INFERSHAPE_FNS_H_