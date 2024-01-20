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
 * \file nn_norm.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief AddRmsNorm operator interface implementation
*  calculating: x1, x2, gamma
*  x = x1 + x2
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon))
*  y = gamma * (x * rstd)

* @par Inputs:
* Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li x2: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li gamma: A Tensor. Must be one of the following types: float16, float32, bfloat16. \n

* @par Attributes:
* @li epsilon: A optional attribute, the type is float32. Defaults to 1e-6 . \n

* @par Outputs:
* Two outputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li rstd: A Tensor. Must be one of the following types: float32.
* @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
*/
REG_OP(AddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6)
    .OP_END_FACTORY_REG(AddRmsNorm)

}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_
