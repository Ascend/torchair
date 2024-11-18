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
 * \file nn_quantize.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Function DynamicQuant.

* @par Inputs:
* three input, including:
* @li x: A matrix Tensor, to be quantized. The type support float16, bfloat16.
* @li smooth_scales: A matrix Tensor, to smooth x before quantization. The type support float16, bfloat16. \n
* @li group_index: A matrix Tensor, to select the smooth_scale for different rows in x. The type support int32. \n

* @par Attributes:
* one attr, including:
* @li dst_type: a int32 scalar, specifying the output data type.
* than 0.

* @par Outputs:
* two outputs, including:
* y: A matrix Tensor, quantized from x. The type support int8.
* scale: A matrix Tensor, scale factor used in quantization. The type support float32.
*/
REG_OP(DynamicQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(DynamicQuant)


/**
* @brief Function DynamicQuantV2.

* @par Inputs:
* three input, including:
* @li x: A matrix Tensor, to be quantized. The type support float16, bfloat16.
* @li smooth_scales: A matrix Tensor, to smooth x before quantization. The type support float16, bfloat16. \n
* @li group_index: A matrix Tensor, to select the smooth_scale for different rows in x. The type support int32. \n

* @par Attributes:
* one attr, including:
* @li dst_type: a int32 scalar, specifying the output data type.
* than 0.

* @par Outputs:
* three outputs, including:
* y: A matrix Tensor, quantized from x. The type support int8.
* scale: A matrix Tensor, scale factor used in quantization. The type support float32.
* offset: A matrix Tensor, offset factor used in quantization. The type support float32.
*/
REG_OP(DynamicQuantV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(offset, TensorType({DT_FLOAT}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(DynamicQuantV2)

} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
