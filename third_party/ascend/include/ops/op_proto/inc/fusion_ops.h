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
 * \file fusion_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Function MatmulAllReduce.

* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bf16, int8.
* @li x2: A matrix Tensor. The type support float16, bf16, int8, int4.
* @li bias: A matrix Tensor. The type support float16, bf16, int32.
* @li x3: A matrix Tensor. The type support float16, bf16, float16.
* @li antiquant_scale: A matrix Tensor. The type support float16, bf16, float16.
* @li antiquant_offset: A matrix Tensor. The type support float16, bf16, float16.
* @li dequant_scale: A matrix Tensor. The type support float32, float16, bf16, uint64.
* @li pertoken_scale: A matrix Tensor. The type support float32.
* @li comm_quant_scale_1: A matrix Tensor. The type support float16, bf16.
* @li comm_quant_scale_2: A matrix Tensor. The type support float16, bf16. \n

* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
 perform. support "sum", "min", "max" ,"prod" .
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. Default: false.
* @li comm_turn: A int. Number of communications with AICPU. Default: 0.
* @li antiquant_group_size: A int. For per-group. Default: 0. \n

* @par Outputs:
* y: A matrix Tensor. The type support float16, bf16, float16.
*/
REG_OP(MatmulAllReduce)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT64, DT_INT64}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(comm_quant_scale_1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(comm_quant_scale_2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .OP_END_FACTORY_REG(MatmulAllReduce)

/**
* @brief Fusion op of allgather and matmul.
* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bfloat16.
* @li x2: A matrix Tensor. The type support float16, bfloat16.
* @li bias: A matrix Tensor. The type support float16, bfloat16. \n
*
* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication. Default: "false".
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. Default: "false".
* @li gather_index: A int. Represents the input index for doing gather.
  Default: "0".
* @li comm_turn: A int. Number of communications with AICPU. Default: "0". \n
*
* @par Outputs:
* @li y: A matrix Tensor. The type support float16, bfloat16.
* @li gather_out: A matrix Tensor. The type support float16, bfloat16. \n
*/
REG_OP(AllGatherMatmul)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(gather_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(gather_index, Int, 0)
    .ATTR(comm_turn, Int, 0)
    .OP_END_FACTORY_REG(AllGatherMatmul)

/**
* @brief Function WeightQuantBatchMatmulV2. \n

* @par Inputs:
* @li x: A matrix Tensor.
* @li weight: A matrix Tensor of quantized weight.
* @li antiquant_scale: A Tensor for antiquant scale.
* @li antiquant_offset: A Tensor for antiquant offset.
* @li quant_scale: A Tensor for quantization parameters.
* @li quant_offset: A Tensor for quantization parameters.
* @li bias: A Tensor. \n


* @par Attributes:
* @li transpose_x: A bool. x is transposed if true.
* @li transpose_weight: A bool. weight is transposed if true. \n

* @par Outputs:
* y: A matrix Tensor.
*/
REG_OP(WeightQuantBatchMatmulV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_INT8, DT_INT4}))
    .INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT, DT_UINT64}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(transpose_x, Bool, false)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(dtype, Int, -1)
    .OP_END_FACTORY_REG(WeightQuantBatchMatmulV2)

/**
* @brief Combine similar tokens using the matching algorithm.
* @par Inputs:
* @li token_a: A Tensor. Type is:DT_FLOAT16.
* @li token_b: A Tensor. Type is:DT_FLOAT16.
* @li topk_indice: A Tensor. Type is:DT_INT64.
* @li arg_max: A Tensor. Type is:DT_INT64.
* @par Outputs:
* @li unmerge_token_a: A Tensor. Type is:DT_FLOAT16.
* @li unmerge_token_b: A Tensor. Type is:DT_FLOAT16.
* @li unreduce_count: A Tensor. Type is:DT_FLOAT.
* @par Attributes:
* @li top_rate: Type is:Float.
*/
REG_OP(TomeMerge)
    .INPUT(token_a, TensorType({DT_FLOAT16}))
    .INPUT(token_b, TensorType({DT_FLOAT16}))
    .INPUT(topk_indice, TensorType({DT_INT64}))
    .INPUT(arg_max, TensorType({DT_INT64}))
    .OUTPUT(unmerge_token_a, TensorType({DT_FLOAT16}))
    .OUTPUT(unreduce_token_b, TensorType({DT_FLOAT16}))
    .OUTPUT(unreduce_count, TensorType({DT_FLOAT}))
    .ATTR(top_rate, Float, 0.5)
    .OP_END_FACTORY_REG(TomeMerge)

/**
 * @brief TomeUnmerge.
 * @par Inputs:
 * thirteen input, including:
 * @li atten_out: A Tensor. Support float16
 * @li Ori_IndiceA: A Tensor. Support int64
 * @li Ori_IndiceB: A Tensor. Support int64
 * @li TOPK_Indice: A Tensor. Support int64
 * @li Arg_Max: A Tensor. Support float16 \n

 * @par Attributes:
 * @li topRRate: Support float
 * @li attn_dim_per_head: Attention dim of a Head, Support int\n

 * @par Outputs:
 * Eight output, including:
 * @li unZipToken: A Tensor. Result of Attention. Support float16

 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(TomeUnmerge)
    .INPUT(atten_out, TensorType({DT_FLOAT16}))
    .INPUT(Ori_IndiceA, TensorType({DT_INT64}))
    .INPUT(Ori_IndiceB, TensorType({DT_INT64}))
    .INPUT(TOPK_Indice, TensorType({DT_INT64}))
    .INPUT(Arg_Max, TensorType({DT_FLOAT16}))
    .OUTPUT(unZipToken, TensorType({DT_FLOAT16}))
    .ATTR(top_r_rate, Float, 0.5)
    .OP_END_FACTORY_REG(TomeUnmerge)

/**
* @brief Fusion op of matmul and reduce scatter.
* @par Inputs:
* twelve inputs, including:
* @li x1: A matrix Tensor. The type support float16, bfloat16.
* @li x2: A matrix Tensor. The type support float16, bfloat16.
* @li bias: A matrix Tensor. The type support float16, bfloat16. \n
*
* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
  perform.
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication. Default: "false".
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. Default: "false".
* @li comm_turn: A int. Number of communications with AICPU. Default: "0". \n
*
* @par Outputs:
* y: A matrix Tensor. The type support float16, bfloat16. \n
*/
REG_OP(MatmulReduceScatter)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .OP_END_FACTORY_REG(MatmulReduceScatter)

/**
* @brief Quant Batch Matmul Version 3 Calculation.

* @par Inputs:
* Six inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: int8, int4.
          The format supports ND. The shape should be within 2 ~ 6 dimension.
          When transpose_x1 is false, the shape is (batch,m,k), where batch is optional.
* @li x2: A matrix Tensor. Must be one of the following types: int8, the format supports ND and NZ;
          int4, the format only supports ND.
          In ND format and int8 dtype, the shape ranges from 2D to 6D. When transpose_x2 is false, the shape is (batch,k,n), where
          batch is optional; in int4 dtype, shape only supports 2D.
          In NZ (Ascend affinity) format, the shape ranges from 4D to 8D. When tranpose_x2 is true, the shape is
          (batch,k1,n1,n0,k0), batch is optional, k0 = 32, and n0 = 16. k in the shape of x1 and k1 in the shape of x2
          must meet the following requirement: ceilDiv(k,32) = k1. When transpose_x2 is false, the shape is
          (batch,n1,k1,k0,n0), batch is optional, k0 = 16, n0 = 32. k in the shape of x1 and k1 in the shape of x2 must
          meet the following requirement: ceilDiv(k,16) = k1.
* @li scale: A matrix Tensor, quantization parameter.
             Must be one of the following types: uint64, float32, int64, bfloat16. The format
             supports ND. The shape is 1D (t,), with t equal to 1 or n, where n is the same as that of x2.
             When the output is int8 type, data type of scale have to be int64 or uint64.
             When the output is bfloat16 type, data type of scale have to be bfloat16 or float32.
             When the outout is float16 type, if pertoken_scale is not empty, scale have to be float32 type.
* @li offset: An optional matrix Tensor, quantization parameter. Must be one of the following types: float32.
              The format supports ND. The shape is 1D (t,), with t equal to 1 or n, where n is the same as that of x2.
* @li bias: An optional matrix Tensor. Must be one of the following types: int32, bfloat16, float16, float32. The format supports ND.
            The shape is 1D (t,) or 3 dimensional(batch, 1, n),
            with t equal to n, where n is the same as that of x2.
* @li pertoken_scale: A optional matrix Tensor. The type supports float32. The format supports ND.
                      The shape is 1D (t,), with t equal to m, where m is the same as that of x1. \n

* @par Attributes:
* Three attributes, including:
* @li dtype: A Int. Declare the output dtype, supports 1(float16), 2(int8), 27(bfloat16). Default: 2(int8).
* @li transpose_x1: A bool. If true, changes the shape of "x1" from [m, k] to
* [k, m] before multiplication. Default: false.
* @li transpose_x2: A bool. If true, changes the shape of "x2" from [k, n] to
* [n, k] before multiplication. Default: false. \n

* @par Outputs:
* One output, including:
* y: A matrix Tensor. Must be one of the following types: float16, int8, bfloat16, int32.
     The format supports ND. The shape ranges from 2D to 6D,
     that is, (batch,m,n), where batch is optional. Broadcasting can be performed on the batch dimension of x1 and x2.
     The output batch is the same as the batch after broadcasting, m is the same as that of x1, and n is the same as
     that of x2. \n

* @attention Constraints:
* @li The shape of bias should be 1D when the shape of out is 2D, 4D, 5D or 6D, and the shape of bias should be 1D or 3D
* when the out shape is 3D.
* @li The size of the last dimension of x1 or x2 cannot exceed 65535. The last dimension of x1 refers to m when
* transpose_x1 is true or k when transpose_x1 is false. The last dimension of x2 refers to k when transpose_x2 is true
* or n when transpose_x2 is false.
* @li If input dtype of x1 and x2 is int4, transpose_x1 should be false, the size of the last dimension of x1 or x2 should
* be an even number.
* @li Input does not support tensor with dimension size 0.
* @li The following are the supported data type combinations by platform.

* - Atlas Inference Series Product:
*\n
| x1       | x2       | scale        | offset   | bias     | pertoken | out      |
| -------: | :------: | :----------: | :------: | :------: | :------: | :------: |
| int8     | int8     | uint64/int64 | null     | int32    | null     | float16  |
| int8     | int8     | uint64/int64 | float32  | int32    | null     | int8     |
*\n
* - Atlas A2 Trainging Series Product/Atlas 800I A2 Inference Product or Atlas A3 Training Series Product:
*\n
| x1       | x2       | scale            | offset   | bias                   | pertoken     | out      |
| -------: | :------: | :--------------: | :------: | :--------------------: | :----------: | :------: |
| int8     | int8     | uint64/int64     | null     | int32                  | null         | float16  |
| int8     | int8     | uint64/int64     | float32  | int32                  | null         | int8     |
| int8     | int8     | float32/bfloat16 | null     | int32/bfloat16/float32 | null/float32 | bfloat16 |
| int8     | int8     | float32          | null     | int32/float16/float32  | float32      | float16  |
| int4     | int4     | uint64/int64     | null     | int32                  | null         | float16  |
| int8     | int8     | float32/bfloat16 | null     | int32                  | null         | int32    |
*\n
*/
REG_OP(QuantBatchMatmulV3)
    .INPUT(x1, TensorType({DT_INT8, DT_INT4}))
    .INPUT(x2, TensorType({DT_INT8, DT_INT4}))
    .INPUT(scale, TensorType({DT_UINT64, DT_FLOAT, DT_INT64, DT_BF16}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32, DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT8, DT_BF16, DT_INT32}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(QuantBatchMatmulV3)

/**
* @brief Function TransQuantParamV2.

* @par Inputs:
* two inputs, including:
* @li scale: A matrix Tensor. The type support uint64, int64.
* @li offset: A matrix Tensor. The type support float32.

* @par Outputs:
* y: A matrix Tensor. The type support int64.
*/
REG_OP(TransQuantParamV2)
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(TransQuantParamV2)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
