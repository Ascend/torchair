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
* @brief Fusion op DequantRopeQuantKvcache.

* @par Inputs:
* thirteen inputs, including:
* @li x: A Tensor with shape (B, S, H) or (B, H), H is (Nq+Nkv+Nkv)*D, format support ND.
* The type support float16, bf16, int32.
* @li cos: A Tensor with shape (B, S, 1, D) or (B, D). The type support float16, bf16, format support ND.
* @li sin: A Tensor with shape (B, S, 1, D) or (B, D). The type support float16, bf16, format support ND.
* @li k_cache: A Tensor with shape (C_1, C_2, Nkv, D) indicates kcache for in-place updates.
* The type support int8, format support ND.
* @li v_cache: A Tensor with shape (C_1, C_2, Nkv, D) indicates vcache for in-place updates.
* The type support int8, format support ND.
* @li indices: A Tensor with shape (B) when cache_mode is contiguous with shape (B * S) when cache_mode is page.
* The type support int32, format support ND.
* @li scale_k: A Tensor with shape (Nkv, D). The type support float32, format support ND.
* @li scale_v: A Tensor with shape (Nkv, D). The type support float32, format support ND.
* @li offset_k: A Tensor with shape (Nkv, D). An optional input parameter. The type support float32.
* format support ND.
* @li offset_v: A Tensor with shape (Nkv, D). An optional input parameter. The type support float32.
* format support ND.
* @li weight_scale: A Tensor with shape (D) indicates the weight scale factor of the dequantization parameter.
* An optional input parameter. The type support float32.
* @li activation_scale: A Tensor with shape (B * S) or (B) indicates the activation scale factor of the dequantization parameter.
* An optional input parameter. The type support float32.
* @li bias: A Tensor with shape (D). An optional input parameter. The type support float32, bf16, float16, int32.

* @par Attributes:
* @li size_splits: A list of int. Specifies the size of spliting qkv.
* @li quant_mode: A string. A optional attribute. Specifies the method of quant. Default: "static".
* @li layout: A string. A optional attribute. Specifies the format of input. Default: "BSND".
* @li kv_output: A bool. A optional attribute. Whether to output kv. Default: "false".
* @li cache_mode:  A string. A optional attribute. Specifies the cache mode for kcache and vcache.
*    Should be "contiguous" or "page", default is "contiguous".

* @par Outputs:
* @li q: A Tensor with shape (B, S, Nq, D) or (B, Nq, D). The type support float16, bf16.
* @li k: A Tensor with shape (B, S, Nkv, D) or (B, Nkv, D). The type support float16, bf16.
* @li v: A Tensor with shape (B, S, Nkv, D) or (B, Nkv, D). The type support float16, bf16.
* @li k_cache: A Tensor with shape (C_1, C_2, Nkv, D). The type support int8, format support ND.
* @li v_cache: A Tensor with shape (C_1, C_2, Nkv, D). The type support int8, format support ND.
*/
REG_OP(DequantRopeQuantKvcache)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(k_cache, TensorType({DT_INT8}))
    .INPUT(v_cache, TensorType({DT_INT8}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(scale_k, TensorType({DT_FLOAT32}))
    .INPUT(scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(offset_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(offset_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT32, DT_BF16, DT_FLOAT16, DT_INT32}))
    .OUTPUT(q, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(k, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(v, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(k_cache, TensorType({DT_INT8}))
    .OUTPUT(v_cache, TensorType({DT_INT8}))
    .REQUIRED_ATTR(size_splits, ListInt)
    .ATTR(quant_mode, String, "static")
    .ATTR(layout, String, "BSND")
    .ATTR(kv_output, Bool, false)
    .ATTR(cache_mode, String, "contiguous")
    .OP_END_FACTORY_REG(DequantRopeQuantKvcache)

/**
* @brief DequantBias. \n

* @par Inputs:
* @li x: A tensor of type int32.
* @li weight_scale: A tensor of type float or bf16.
* @li activate_scale: A tensor of type float.
* @li bias: A tensor of type float, bf16, float16 or int32.

* @par Attributes:
* @li output_dtype: A int attr.

* @par Outputs:
* @li y: A tensor of type int32. \n
*/
REG_OP(DequantBias)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(weight_scale, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(activate_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32, DT_INT32}))
    .REQUIRED_ATTR(output_dtype, Int)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(DequantBias)

/**
* @brief AddLora, the operation kernel for batch gather matmul.
* @par Inputs:
* @li y: A tensor of type float16. Supported format "ND".
* @li x: A tensor of type float16. Supported format "ND".
* @li weightB: A weight tensor of type float16. Supported format "ND". Represents the second weight matrix for matrix multiplication.
* @li indices: A tensor of type int32. Supported format "ND". The shape must be the same as the first dim of x. Identifies the group index of the input x.
* @li weightA: A optional weight tensor of type float16. Supported format "ND". Represents the first weight matrix for matrix multiplication. If empty, the first matrix multiplication will be skipped.

* @par Attributes:
* @li layer_idx: A optional int, default value is 0, indicates the layer id of weight tensors.
* @li scale: A optional float, default value is 1e-3, scales up the multiplication results.
* @li y_offset: A optional int, default value is 0, represents the offset of y.
* @li y_slice_size: A optional int, default value is -1, represents the slice_size of y to be updated.

* @par Outputs:
* y_out: A tensor of type float16, the shape requirements are consistent with the shape of y.
*/
REG_OP(AddLora)
    .INPUT(y, TensorType({DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(weightB, TensorType({DT_FLOAT16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(weightA, TensorType({DT_FLOAT16}))
    .ATTR(layer_idx, Int, 0)
    .ATTR(scale, Float, 1e-3)
    .ATTR(y_offset, Int, 0)
    .ATTR(y_slice_size, Int, -1)
    .OUTPUT(y_out, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(AddLora)

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

* @par Attributes:
* @li round_mode: fp32 filing to fp19 mode; 0: truncation and filing; 1: r_int mode;

* @par Outputs:
* y: A matrix Tensor. The type support int64.
*/
REG_OP(TransQuantParamV2)
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .ATTR(round_mode, Int, 0)
    .OP_END_FACTORY_REG(TransQuantParamV2)

/**
* @brief Function GroupedMatmulFinalizeRouting. This op mixs GroupedMatmul和MoeFinalizeRouting. After the calculation of GroupedMatmul, perform a combine operation on the output according to the index, and support the format where w is in the Ascend affinity data layout.
 
* @par Inputs:
* @li x: A tensor, which is the input x in the formula, supports the ND data format (refer to Data Format), and the shape supports 2 dimensions with the dimension being (m, k). The data type supports INT8.

* @li w: A tensor of weight, which supports the Ascend affinity data layout format as described in Data Format. The data type supports INT8, INT4, and the shape supports 5 dimensions.
* When transposew is false, each dimension is represented as: (e, n1, k1, k0, n0), where k0 = 16, n0 = 32. The k in the x shape and the k1 in the w shape need to satisfy the following relationship: ceilDiv(k, 16) = k1.
* The aclnnCalculateMatmulWeightSizeV2 interface and the aclnnTransMatmulWeight interface can be used to complete the conversion of the input format from the ND format to the Ascend affinity data layout format.
* @li scale: It represents the scaling factor in the quantization parameters. It supports the ND data format as described in Data Format. The supported data type is FLOAT32, INT64
* and if w is INT4, the shape is three-dimensional (e, 1, n), others scenarios is two-dimensional (e, n), where the values of e and n are consistent with those of e and n in w.
* @li bias: A tensor of bias, contains all bias of inputs for matmul. For each tensor, the data type of elements supports float32; the format supports ND. Currently, input is not supported.
* @li offset: A tensor of offset, only supported when w is INT4, the shape is three-dimensional (e, 1, n), the format supports ND. The supported data types is FLOAT32.
* @li pertoken_scale: The dequantization parameters for matrix calculation support the ND data format (refer to Data Format). They correspond to the x matrix, with a dimension of (m). The supported data type is FLOAT32. Non - contiguous tensors are not supported.
* @li group_list: a tensor, indicates M-axis distributation of groups of matmuls for inputs and outputs.
* Data type of elements is int64. Format: ND.
* @li shared_input: In the MoE (Mixture of Experts) calculation, the output of the shared experts needs to undergo a combine operation with the output of the MoE experts. The supported data types are bfloat16.
* @li logit: In the MoE (Mixture of Experts), for the logit magnitudes of each token, the output of the matrix multiplication is multiplied by these logits, and then combined according to the indices. The supported data type is float32.
* @li row_index: The outputs of the MoE (Mixture of Experts) are combined according to the rowIndex, where the values in rowIndex serve as the indices for the scatter add operation during the combination. The supported data types are int64 and int32.
* @par Attributes:
* @li dtype: The type of GroupedMatmul. The type is int, which output:0：FLOAT32；1：FLOAT16；2：BFLOAT16.
* @li shared_input_weight: The coefficients for combining the shared experts and the MoE experts. The shareInput is multiplied first with these coefficients, and then the result is accumulated with the output of the MoE experts. The supported data type is float32.
* @li shared_input_offset: The offset of the output of the shared experts in the total output. The supported data type is int64.
* @li transpose_x: Whether the left matrix is transposed. Default value: false(not transposed).
* @li transpose_w: Whether the right matrix is transposed. Default value: false(not transposed).
* @li output_bs: The size of the highest dimension of the output.
* @li group_list_type: Group type of GroupedMatmul. Default value: 1. When configured as 0: It is in the cumsum mode, which means it is the prefix sum. When configured as 1: It is in the count mode. The supported data type is int64. \n

* @par Outputs:
* y: A tensor List, contains all result of groups of matmuls. For each tensor,
* the data type of elements supports float32; the format supports ND. \n

* @attention Constraints:
* Support combinations of data type: \n
* | x   | w   | Scale      | Scale          | pertokenScale      | bias                        | out      | \n
*  | ---- | ---- | ------------ | ---------------- | ------------- | --------------------------- | -------- | \n
*  | INT8 | INT8 | null         | UINT64     | null          | null/INT32                  | FLOAT16  | \n
*  | INT8 | INT8 | null         | UINT64     | null/FLOAT32  | null/INT32                  | INT8     | \n
* Among them: \n
* The dimension m = batch * topk, and the value range is [1, 16 * 1024 * 8]. The functionality is not guaranteed if it exceeds this range. \n
* k supports the value include of 256、512、1024、1408、2048. The functionality is not guaranteed if it exceeds this range. \n
* n supports the value include of 2048、7168、7680. The functionality is not guaranteed if it exceeds this range. \n
* The value range of e is [1, 256]. The functionality is not guaranteed if it exceeds this range. \n
* The value range of bs/p is [1, 2 * 1024], and p = [8, 16, 32, 48, 64, 96, 128, 144, 288]. \n
* The value range of bs is [1, 16 * 1024]. The functionality is not guaranteed if it exceeds this range. \n
* The sum of the values in grouplist is less than or equal to m. \n
*/
REG_OP(GroupedMatmulFinalizeRouting)
.INPUT(x, TensorType({DT_INT8}))
.INPUT(w, TensorType({DT_INT8, DT_INT4}))
.OPTIONAL_INPUT(scale, TensorType({DT_FLOAT, DT_INT64}))
.OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
.OPTIONAL_INPUT(shared_input, TensorType({DT_BF16}))
.OPTIONAL_INPUT(logit, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(row_index, TensorType({DT_INT64, DT_INT32}))
.OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
.OUTPUT(y, TensorType({DT_FLOAT}))
.ATTR(dtype, Int, 0)
.ATTR(shared_input_weight, Float, 1.0)
.ATTR(shared_input_offset, Int, 0)
.ATTR(transpose_x, Bool, false)
.ATTR(transpose_w, Bool, false)
.ATTR(output_bs, Int, 0)
.ATTR(group_list_type, Int, 1)
.OP_END_FACTORY_REG(GroupedMatmulFinalizeRouting)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
