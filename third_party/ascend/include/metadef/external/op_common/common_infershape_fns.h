/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

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
