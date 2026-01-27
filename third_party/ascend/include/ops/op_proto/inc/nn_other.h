/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file nn_other.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Apply rotary position embedding.
 * @par Inputs:
 * x: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r1: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r2: A tensor. Must be one of the following types: float16, float, bfloat16. 
 * @par Outputs:
 * y: A Tensor. Has the same shape as "x".
 */
REG_OP(RotaryMul)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(r1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(r2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(RotaryMul)

/**
 * @brief Calculate the inverse gradient of RotaryMul.
 * @par Inputs:
 * x: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r1: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r2: A tensor. Must be one of the following types: float16, float, bfloat16.
 * dy: A tensor. Data of grad increment.
 * @par Outputs:
 * dx: A Tensor. Has the same shape as "x".
 * dr1: A Tensor. Has the same shape as "r1".
 * dr2: A Tensor. Has the same shape as "r2".
 */
REG_OP(RotaryMulGrad)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(r1, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(r2, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OUTPUT(dr1, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OUTPUT(dr2, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(RotaryMulGrad)

/**
 * @brief Apply rotary position embedding for a single tensor.
 * @par Inputs:
 * @li x: A 4D tensor which rotary position embedding is applied, format supports ND, and data type must be float16, float or bfloat16.
 * @li cos: A 4D tensor which is "cos" in rotary position embedding, format supports ND, data type must be the same as "x", and shape must be the same as "sin".
 * @li sin: A 4D tensor which is "sin" in rotary position embedding, format supports ND, data type must be the same as "x", and shape must be the same as "cos".
 * @par Outputs:
 * y: A 4D tensor which is the result of rotary position embedding, format supports ND, data type must be the same as "x", and shape must be the same as "x".
 * @par Attributes:
 * mode: An optional attribute of type int, specifying the mode of rotary position embedding, must be 0-"half", 1-"interleave", 2-"quarter" or 3-"interleave-half". Defaults to 0.
 * Atlas A2 Training Series Product/ Atlas 800I A2 Inference Product and Atlas A3 Training Series Product only support 0-"half" and 1-"interleave".
 * @attention Constraints:
 * Let (B, S, N, D) represents the shape of the 4-D input "x". Under this representation, the shape constraints of each parameter can be described as follows:
 * @li The D of "x", "cos", "sin" and "y" must be equal. For Ascend 910_95 AI Processor, D should be less or equal to 1024. 
 * For Atlas A2 Training Series Product/ Atlas 800I A2 Inference Product and Atlas A3 Training Series Product, D should be less or equal to 896.
 * @li In half, interleave and interleave-half mode, D must be a multiple of 2. In quarter mode, D must be a multiple of 4.
 * @li B, S, N of "cos" and "sin" must meet one of the following four conditions:
 *  - B, S, N are 1, means the shape is (1, 1, 1, D).
 *  - B, S, N are the same as that of "x", means the shape is (B, S, N, D).
 *  - One of S and N is 1, the remaining one dimension and B are the same as that of "x", means the shape is (B, 1, N, D) or (B, S, 1, D).
 *  - Two of B, S and N are 1, the remaining one dimension is the same as that of "x", means the shape is (1, 1, N, D), (1, S, 1, D) or (B, 1, 1, D).
 */
REG_OP(RotaryPositionEmbedding)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(mode, Int, 0)
    .OP_END_FACTORY_REG(RotaryPositionEmbedding)

/**
 * @brief Backwards calculation of RotaryPositionEmbedding.
 * @par Inputs:
 * @li dy: A 4D tensor which represents the gradient of output "y" in RotaryPositionEmbedding, format supports ND, and data type must be float16, float or bfloat16.
 * @li cos: A 4D tensor which is input "cos" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "sin".
 * @li sin: A 4D tensor which is input "sin" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "cos".
 * @li x: An optional 4D tensor which is input "x" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "dy".
 * If "x" is nullptr, the output "dcos" and "dsin" is meaningless.
 * @par Outputs:
 * @li dx: A 4D Tensor which is the grad of input "x" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "dy".
 * @li dcos: A 4D Tensor which is the grad of input "cos" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "cos".
 * @li dsin: A 4D Tensor which is the grad of input "sin" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "sin".
 * @par Attributes:
 * mode: An optional attribute of type int, specifying the mode of rotary position embedding, must be 0-"half", 1-"interleave", 2-"quarter" or 3-"interleave-half". Defaults to 0.
 * Atlas A2 Training Series Product/ Atlas 800I A2 Inference Product and Atlas A3 Training Series Product only support 0-"half" and 1-"interleave".
 * @attention Constraints:
 * Let (B, S, N, D) represents the shape of the 4-D input "dy". Under this representation, the shape constraints of each parameter can be described as follows:
 * @li The D of "dy", "cos", "sin", "x", "dx", "dcos" and "dsin" must be equal. For Ascend 910_95 AI Processor, D should be less or equal to 1024. 
 * For Atlas A2 Training Series Product/ Atlas 800I A2 Inference Product and Atlas A3 Training Series Product, D should be less or equal to 896.
 * @li In half, interleave and interleave-half mode, D must be a multiple of 2. In quarter mode, D must be a multiple of 4.
 * @li B, S, N of "cos", "sin", "dcos" and "dsin" must meet one of the following four conditions:
 *  - B, S, N are 1, means the shape is (1, 1, 1, D).
 *  - B, S, N are the same as that of "dy", means the shape is (B, S, N, D).
 *  - One of S and N is 1, the remaining one dimension and B are the same as that of "dy", means the shape is (B, 1, N, D) or (B, S, 1, D).
 *  - Two of B, S and N are 1, the remaining one dimension is the same as that of "dy", means the shape is (1, 1, N, D), (1, S, 1, D) or (B, 1, 1, D).
 */
REG_OP(RotaryPositionEmbeddingGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dcos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dsin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(mode, Int, 0)
    .OP_END_FACTORY_REG(RotaryPositionEmbeddingGrad)

REG_OP(ApplyRotaryPosEmb)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .ATTR(layout, Int, 1)
    .ATTR(rotary_mode, String, "half")
    .OUTPUT(query, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .OUTPUT(key, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(ApplyRotaryPosEmb)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_
