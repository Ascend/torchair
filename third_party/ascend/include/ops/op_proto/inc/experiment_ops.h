/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file experiment_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Updates "var" according to the AdamW algorithm.
*
* @attention Constraints:
*  The input tensors must have the same shape.*
*
* @par Inputs:
* @li var: A mutable Tensor of the type TensorType::NumberType().
*     Should be from a Variable().
* @li m: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
* @li v: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
* @li beta1_power: A scalar of the same type as "var".
* @li beta2_power: A scalar of the same type as "var".
* @li lr: learning_rate. A scalar of the same type as "var".
* @li weight_decay: learning_rate. A scalar of the same type as "var".
* @li beta1: A scalar of the same type as "var".
* @li beta2: A scalar of the same type as "var".
* @li epsilon: A scalar of the same type as "var".
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li max_grad_norm: A mutable Tensor of the same type as "var", an optional input.
*     Should be from a Variable().
*
* @par Attributes:
* @li amsgrad: An optional bool. Defaults to "False".
*     If "True", max_grad_norm input and output must be entered.
* @li maximize: An optional bool. Defaults to "False".
*
* @par Outputs:
* @li var: A mutable tensor. Has the same type as input "var".
* @li m: A mutable tensor. Has the same type as input "m".
* @li v: A mutable tensor. Has the same type as input "v". \n
*/
REG_OP(ApplyAdamW)
.INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(weight_decay, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OPTIONAL_INPUT(max_grad_norm, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .OUTPUT(v, TensorType::NumberType())
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamW)

/**
* @brief Calculate SQ distance. \n
*
* @par Inputs:
* @li ivf: A Tensor, dtype is uint8.
* @li query: A Tensor, dtype is float16 or float32.
* @li bucket_list: A Tensor, dtype is int32 or int64.
* @li bucket_limits: A Tensor, dtype is int32 or int64.
* @li bucket_offsets: A Tensor, dtype is int32 or int64.
* @li vmin: A Tensor, dtype is float16 or float32.
* @li vdiff: A Tensor, dtype is float16 or float32. \n
*
* @par Outputs:
* @li actual_count: A Tensor, dtype is int32 or int64, the actual number of sq_distance.
* @li sq_distance: A Tensor, dtype is float16 or float32.
* @li grouped_extreme_distance: A Tensor, dtype is float16 or float32, the extremum in each group of sq_distance.
* @li sq_ivf: A Tensor, dtype is int32 or int64.
* @li sq_index: A Tensor, dtype is int32 or int64. \n
*
* @par Attributes:
* @li total_limit: A Int, indicates the max length of the output sq_distance.
* @li group_size: A Int, indicates the group size of the extremum.
* @li extreme_mode: A Int, indicates the type of extremum, 0 means minimum, and 1 means maximum. \n
*
*/
REG_OP(ScanSQCodes)
.INPUT(ivf, TensorType({DT_UINT8}))
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bucket_list, TensorType({DT_INT32, DT_INT64}))
    .INPUT(bucket_limits, TensorType({DT_INT32, DT_INT64}))
    .INPUT(bucket_offsets, TensorType({DT_INT32, DT_INT64}))
    .INPUT(vmin, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(vdiff, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(actual_count, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(sq_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grouped_extreme_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sq_ivf, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(sq_index, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(total_limit, Int)
    .ATTR(group_size, Int, 64)
    .ATTR(extreme_mode, Int, 0)
    .OP_END_FACTORY_REG(ScanSQCodes)

/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b". \n
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float32,
* float16, int32, int8, int4, bf16. 3D. Has format ND.
* @li x2: A matrix Tensor. Must be one of the following types: float32,
* float16, int32, int8, int4, bf16. 3D. Has format ND.
* @li bias: A optional Tensor. Must be one of the following types:
* float32, float16, int32, bf16. 1D. Has format ND.
* @li offset_w: A optional Tensor. Must be one of the following types:
* int8, int4. Has format ND. \n

* @par Attributes:
* Three attributes, including:
* @li perm_x1: A list int. "x1" is permuted to shape [B, M, K] before multiplication.
* @li perm_x2: A list int. "x2" is permuted to shape [B, K, N] before multiplication.
* @li perm_y: A list int. "y" is permuted after multiplication.
* @li offset_x: An optional integer for quantized TransposeBatchMatMul.
* The negative offset added to the input "x1" for int8, int4 type. Ensure offset_x
* within the effective range of input data type. Defaults to "0". \n

* @par Outputs:
* y: The result matrix Tensor. 3D. Must be one of the following
* types: float32, float16, int32, bf16. 3D. Has format ND. \n
*/
REG_OP(TransposeBatchMatMul)
.INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .ATTR(perm_x1, ListInt, {})
    .ATTR(perm_x2, ListInt, {})
    .ATTR(perm_y, ListInt, {})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(TransposeBatchMatMul)

/**
* @brief Performs non-maximum suppression (NMS) on the rotated boxes according
* to their intersection-over-union (IoU). Rotated NMS interatively removes lower
* scoring rotated boxes which have an IoU greater than iou_threshold with
* another (higher scoring) rotated box.

* @par Inputs:
* Three inputs, including:
* @li boxes: A 2D Tensor of float16 or float32 with shape (N, 5). Rotated boxes to
* perform NMS on. They are expected to be in (x1, y1, x2, y2, angle_degress) format.
* @li scores: A 1D Tensor of float16 or float32 with shape (N). Scores for each one of
* the rotated boxes.
* @li labels: A 1D Tensor of int32 or int64 with shape (N). Labels for each one of
* the rotated boxes.

* @par Attributes:
* iou_threshold: A required float attribute. Discards all overlapping rotated
* boxes with IoU < iou_threshold.

* @par Outputs:
* Two outputs, including:
* @li selected_detections: A 2D Tensor of float16 or float32 with shape (N, 5).
* The selected boxes that kept by Rotated NMS, sorted in decreasing order of scores.
* @li keep_indices: A 1D Tensor of int32 or int64 with shape (N). The indices of
* selected_detections.

* @attention Constraints:
* Currently, the tensor type of input (boxes, scores) only support float.
* The tensor type of keep_indices only support int32.
*/
REG_OP(RotatedNMS)
.INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(selected_detections, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(keep_indices, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(iou_threshold, Float)
    .OP_END_FACTORY_REG(RotatedNMS)

/**
* @brief According to the indices, return the value.

* @par Inputs:
* Four inputs, including:
* @li x: A ND Tensor.
* @li indexed_sizes: A 1D Tensor of int64 with shape (N). Sizes for each one of the indexed data.
* @li indexed_strides: A 1D Tensor of int64 with shape (N). Strides for each one of the indexed data.
* @li indices: Dynamic input. A ND Tensor of int64. return the value according to the indices.

* @par Outputs:
* y: The indexed output tensor. Has the same type and format as input "x".
*/
REG_OP(Index)
.INPUT(x, TensorType::BasicType())
    .INPUT(indexed_sizes, TensorType({DT_INT64}))
    .INPUT(indexed_strides, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Index)

/**
* @brief According to the index number of indexes, replace the value
* corresponding to X with the value.

* @par Inputs:
* Five inputs, including:
* @li x: A ND Tensor.
* @li value: A Tensor of the same type as "x".
* @li indexed_sizes: A 1D Tensor of int64 with shape (N). Sizes for each one of the indexed data.
* @li indexed_strides: A 1D Tensor of int64 with shape (N). Strides for each one of the indexed data.
* @li indices: Dynamic input. A Tensor of the indices.

* @par Attributes:
* @li accumulate: Does it support self accumulation. Defaults to false.

* @par Outputs:
* @li x: A Tensor.

* @par Third-party framework compatibility
* Compatible with the Pytorch operator index_put.

* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(IndexPutV2)
.INPUT(x, TensorType::BasicType())
    .INPUT(value, TensorType::BasicType())
    .INPUT(indexed_sizes, TensorType({DT_INT64}))
    .INPUT(indexed_strides, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(x, TensorType::BasicType())
    .ATTR(accumulate, Bool, false)
    .OP_END_FACTORY_REG(IndexPutV2)

/**
* @brief Performs average pooling on the input. Used in the combination of conv + avgpoolupdate to replace avgpool
* @par Inputs:
* x1: Output of upstream Conv2d. A tensor of type float16, float32.
* x2: Input feature map of upstream Conv2d. A tensor of type int8, float16, float32.

* @par Attributes:
* @li ksize: A required list of 4 ints, specifying the size (N, C, H, and W) of the sliding window,
* where N = C = 1, and H and W are positive integers within the range [1, 255].
* @li strides: A required list of 4 ints, specifying the stride of the sliding window.
* The strides of the N and C dimensions are 1.
* The strides of the H and W dimensions are positive integers within the range [1, 63].
* @li padding_mode: A required string, specifying the padding algorithm,
* either "VALID", "SAME" and "CALCULATED".
* With "SAME" means that the outputs will have the same spatial dimensions as its inputs.
* With "VALID" means no padding.
* @li pads: Pad value when padding_mode is "CALCULATED".
* @li data_format: An optional string, specifying the data format of "ksize" and "strides",
* either "NCHW", or "NHWC" (default).
* @li ceil_mode: Use ceil or floor to calculate the output size when padding_mode is "CALCULATED".
* @li exclusive: Ignore padding area or not when calculating average.

* @par Outputs:
* y: The average pooled output tensor. Has the same type and format as input "x1".

* @attention Constraints:
* @li Only single input and single output are supported.
* @li "ksize_H" and "ksize_W" are positive integers within the range [1, 255]. ksize_H * ksize_W < 256
* @li Due to instruction restrictions,
* the values of "strides_h" and "strides_w" are positive integers within the range [1, 63].
* @par Third-party framework compatibility
* Compatible with the TensorFlow/Pytorch/Onnx operator AvgPoolV2.
*/
REG_OP(AvgPoolUpdate)
.INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DA_INT4, DT_INT8, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NHWC")
    .ATTR(ceil_mode, Bool, false)
    .ATTR(exclusive, Bool, true)
    .OP_END_FACTORY_REG(AvgPoolUpdate)

/**
* @brief YUVToRGB

* @par Inputs:
* @li x: A 4-D uint8 Tensor.
*        Must set the format, supported format list ["NYUV"].
* @li matrix: A 1-D float tensor of 2x3x3 elements

* @par Outputs:
* @li y: A 4-D uint8 Tensor.
*        Must set the format, supported format list ["NCHW, NHWC"].

* @par Attributes:
* @li matrix_type: An Int attr, Defaults to 0.
*                  support list [ 0: CSC_MATRIX_BT601_WIDE,
*                                 1: CSC_MATRIX_BT601_NARROW,
*                                 2: CSC_MATRIX_BT709_WIDE,
*                                 3: CSC_MATRIX_BT709_NARROW,
*                                 4: CSC_MATRIX_BT2020_WIDE,
*                                 5: CSC_MATRIX_BT2020_NARROW,
*                                 6: CSC_MATRIX_USR_DEFINE ]
* @li rb_swap: An Int attr, Defaults to 0.
*              support list [ 0: RGB, 1: BGR ]

* @attention Constraints:
* @li Only support in dvpp
*/

REG_OP(YUVToRGB)
.INPUT(x, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(matrix, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .ATTR(matrix_type, Int, 0)
    .ATTR(rb_swap, Int, 0)
    .OP_END_FACTORY_REG(YUVToRGB)

/**
* @brief DecodeJpegPre

* @par Inputs:
* @li contents: A Tensor of type string. 0-D. The JPEG-encoded image.

* @par Outputs:
* @li dvpp_support: indicates if the dvpp support this jpeg image decode.

* @par Attributes:
* @li w_range: An required listInt contains width [min, max].
* @li h_range: An required listInt contains height [min, max].

* @attention Constraints:
* @li Only support in dvpp
*/

REG_OP(DecodeJpegPre)
.INPUT(contents, TensorType({DT_STRING}))
    .OUTPUT(dvpp_support, BOOL)
    .REQUIRED_ATTR(w_range, ListInt)
    .REQUIRED_ATTR(h_range, ListInt)
    .OP_END_FACTORY_REG(DecodeJpegPre)

/**
* @brief init PartitionMap table. \n

* @par Inputs:
* @li ps_num: A Tensor, dtype is int32. 0-D. indicates ps number.
* @li ps_ids: A Tensor, dtype is int32. 1-D. indicates the id of ps. \n

* @par Attributes:
* @li partition_num: A Int, indicates the number of partition. \n
*/
REG_OP(InitPartitionMap)
.INPUT(ps_num, TensorType({DT_INT32}))
    .INPUT(ps_ids, TensorType({DT_INT32}))
    .ATTR(partition_num, Int, 65537)
    .OP_END_FACTORY_REG(InitPartitionMap)

/**
* @brief uninit PartitionMap table. \n
*/
REG_OP(UninitPartitionMap)
.OP_END_FACTORY_REG(UninitPartitionMap)

/**
* @brief init Embedding hashtable. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable. \n

* @par Attributes:
* @li value_total_len: A Int, indicates the length of hashtable value. \n
* @li embedding_dim: A Int, indicates the length of embedding. \n
* @li bucket_size: A Int, Defaults to "0". \n
* @li dtype: A Type for data, Defaults to "DT_FLOAT". \n
* @li initializer_mode: A String of "random_uniform", "truncated_normal" , "constant" or "".
* indicates the algo of init method. Defaults to "".
* @li constant_value: A Float, used when initializer_mode is "constant", Defaults to "0". \n
* @li min: A Float, used when initializer_mode is "truncated_normal", the minimum value of the random number.
* Defaults to "-2".
* @li max: A Float, used when initializer_mode is "truncated_normal", the maximum value of the random number.
* Defaults to "2".
* @li mu: A Float, used when initializer_mode is "truncated_normal", The mean of the truncated_normal.
* Defaults to "0".
* @li sigma: A Float, used when initializer_mode is "truncated_normal", The variance of the truncated_normal.
* Defaults to "1".
* @li seed: A Int, Defaults to "0". \n
* @li seed2: A Int, Defaults to "0". \n
* @li filter_mode: A String of "no_filter" or "counter". indicates the type of the hashmap, Defaults to "no_filter". \n
* @li optimizer_mode: A String of "adam" or "adamw" or "adagrad". indicates the type of the optimizer_mode, Defaults to "".
* @li optimizer_params: Float list, when optimizer_mode is "adagrad", the initialize value of the optimizer. \n
*/
REG_OP(InitEmbeddingHashmap)
.INPUT(table_id, TensorType({DT_INT32}))
    .ATTR(bucket_size, Int, 0)
    .REQUIRED_ATTR(value_total_len, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(dtype, Type, DT_FLOAT)
    .ATTR(initializer_mode, String, "")
    .ATTR(constant_value, Float, 0)
    .ATTR(min, Float, -2)
    .ATTR(max, Float, 2)
    .ATTR(mu, Float, 0)
    .ATTR(sigma, Float, 1)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(filter_mode, String, "no_filter")
    .ATTR(optimizer_mode, String, "")
    .ATTR(optimizer_params, ListFloat, {})
    .OP_END_FACTORY_REG(InitEmbeddingHashmap)

/**
* @brief embedding hsahtable data import. \n

* @par Inputs:
* @li file_path: A Tensor, dtype is string. 0-D. indicates embedding filepath.
* @li ps_id: A Tensor, dtype is int32. 0-D. indicates the id of ps.
* @li table_id: A Tensor, dtype is int32. 1-D. indicates the id of hashtable. \n

* @par Attributes:
* @li embedding_dim: A ListInt. indicates the hashtable value number.
* @li value_total_length: A ListInt. indicates the hashtable total length, inclue m+v or accum.
* @li only_var: A Bool. only import var.
* @li file_type: A String. indicates the import file .
* @li table_name: A List String. represents table name corresponding to table id . \n
*/
REG_OP(EmbeddingTableImport)
.INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .REQUIRED_ATTR(value_total_len, ListInt)
    .ATTR(only_var_flag, Bool, false)
    .ATTR(file_type, String, "bin")
    .ATTR(table_name, ListString, {})
    .OP_END_FACTORY_REG(EmbeddingTableImport)

/**
* @brief embedding hsahtable data lookup. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable.
* @li keys: A Tensor, dtype is uint32. 1-D. indicates the hashtable key. \n

* @par Outputs:
* @li values: indicates the hashtable value. \n

* @par Attributes:
* @li embedding_dim: A Int. indicates the hashtable value number. \n
*/
REG_OP(EmbeddingTableFind)
.INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(EmbeddingTableFind)

/**
* @brief uninit embedding hsahtable. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable. \n
*/
REG_OP(UninitEmbeddingHashmap)
.INPUT(table_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(UninitEmbeddingHashmap)

/**
* @brief embedding hashtable lookup or init. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is DT_INT32. 0-D. indicates the id of hashtable.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key. \n

* @par Outputs:
* @li values: indicates the hashtable value. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding var value in hashtable.
* @li value_total_len: A Int, indicates the dim of embedding var+m+v or var+accum values in hashtable
* @li initializer_mode: A String of "random_uniform", "truncated_normal" or "constant".
* indicates the algo of init method, Defaults to "random_uniform".
* @li constant_value: A Float, used when initializer_mode is "constant", Defaults to "0".
* @li min: A Float, used when initializer_mode is "truncated_normal", the minimum value of the random number.
* Defaults to "-2".
* @li max: A Float, used when initializer_mode is "truncated_normal", the maximum value of the random number.
* Defaults to "2".
* @li mu: A Float, used when initializer_mode is "truncated_normal", The mean of the truncated_normal.
* Defaults to "0".
* @li sigma: A Float, used when initializer_mode is "truncated_normal", The variance of the truncated_normal.
* Defaults to "1".
* @li seed: An Int, Used to create a random seed, Defaults to "0".
* @li seed2: An Int, Used to create a random seed, Defaults to "0".
* @li filter_mode: A String of "no_filter" or "counter". indicates the type of the hashmap, Defaults to "no_filter".
* @li filter_freq: An Int, Used to set the threshold of the tal, Defaults to "0".
* @li default_key_or_value: A bool, indicates the default value get way.
* @li default_key: An Int, when default_key_or_value is true, use the default_key corresponding value as default value.
* @li default_value: An Int, when default_key_or_value is false, use the default_value as default value.
* @li optimizer_mode: A String of "adam" or "adamw" or "adagrad". indicates the type of the optimizer_mode, Defaults to "".
* @li optimizer_params: Float list, when optimizer_mode is "adagrad", the initialize value of the optimizer. \n
*/
REG_OP(EmbeddingTableFindAndInit)
.INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .REQUIRED_ATTR(value_total_len, Int)
    .ATTR(initializer_mode, String, "random_uniform")
    .ATTR(constant_value, Float, 0)
    .ATTR(min, Float, -2)
    .ATTR(max, Float, 2)
    .ATTR(mu, Float, 0)
    .ATTR(sigma, Float, 1)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(filter_mode, String, "no_filter")
    .ATTR(filter_freq, Int, 0)
    .ATTR(default_key_or_value, Bool, false)
    .ATTR(default_key, Int, 0)
    .ATTR(default_value, Float, 0)
    .ATTR(optimizer_mode, String, "")
    .ATTR(optimizer_params, ListFloat, {})
    .OP_END_FACTORY_REG(EmbeddingTableFindAndInit)

/**
* @brief embedding hashtable embedding applyadam. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li beta1_power: A Scalar, dtype is DT_FLOAT16 or DT_FLOAT. 0-D. indicates the beta1's power.
* @li beta2_power: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta2's power.
* @li lr: A Scalar, dtype is same as "beta1_power". 0-D. indicates the learning rate.
* @li beta1: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta1 param.
* @li beta2: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta2 param.
* @li epsilon: A Scalar, dtype is same as "beta1_power". 0-D. indicates the small value param.
* @li grad: A Tensor, dtype is same as "beta1_power". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable. \n
*/
REG_OP(EmbeddingApplyAdam)
.INPUT(var_handle, TensorType({DT_RESOURCE}))
    .INPUT(beta1_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(epsilon, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(EmbeddingApplyAdam)

/**
* @brief embedding hashtable embedding applyadamW. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li beta1_power: A Tensor, dtype is float16 or float. 0-D. indicates the beta1's power.
* @li beta2_power: A Tensor, dtype is same as "beta1_power". 0-D. indicates the beta2's power.
* @li lr: A Tensor, dtype is same as "beta1_power". 0-D. indicates the learning rate.
* @li weight_decay: A Tensor, dtype is same as "beta1_power". 0-D. indicates the weight decay.
* @li beta1: A Tensor, dtype is same as "beta1_power". 0-D. indicates the beta1 param.
* @li beta2: A Tensor, dtype is same as "beta1_power". 0-D. indicates the beta2 param.
* @li epsilon: A Tensor, dtype is same as "beta1_power". 0-D. indicates the small value param.
* @li grad: A Tensor, dtype is same as "beta1_power". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is int64. 1-D. indicates the hashtable key.
* @li max_grad_norm: A mutable Tensor of the same type as "beta1_power", an optional input. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable.
* @li amsgrad: An optional bool, indicates whether to use the AMSGrad variant of htis algorithm from
*     the paper On the Convergence of Adam and Beyond(default:False).
*     If "True", max_grad_norm input and output must be entered.
* @li maximize: An optional bool, maximize the params based on the objective(default:False). \n
*/
REG_OP(EmbeddingApplyAdamW)
.INPUT(var_handle, TensorType({DT_RESOURCE}))
    .INPUT(beta1_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(weight_decay, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(epsilon, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(max_grad_norm, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingApplyAdamW)
/**
* @brief embedding hashtable export. \n

* @par Inputs:
* @li file_path: A String, indicates the export file path.
* @li ps_id: A Int, dtype is DT_INT32, indicates the ps server id.
* @li table_id: A Tensor, 1D, dtype is DT_INT32, indicates the hashtable id.

* @par Attributes:
* @li embedding_dim: A ListInt. indicates the hashtable value number.
* @li value_total_length: A ListInt. indicates the hashtable total length, inclue m+v or accum.
* @li export_mode: A String. export mode, Defaults to "all".
* @li only_var: A Bool. only export var, Defaults to "false".
* @li file_type: A String. indicates the export file, Defaults to "bin".
* @li table_name: A List String. represents table name corresponding to table id .
* @li filter_export_flag: A Bool. represents filter export flag on counter filter scenario. \n
*/
REG_OP(EmbeddingTableExport)
.INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .REQUIRED_ATTR(value_total_len, ListInt)
    .ATTR(export_mode, String, "all")
    .ATTR(only_var_flag, Bool, false)
    .ATTR(file_type, String, "bin")
    .ATTR(table_name, ListString, {})
    .ATTR(filter_export_flag, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingTableExport)

/**
* @brief embedding tableid trans to resource. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable.

* @par Outputs:
* @li table_handle: indicates the resource_handle of tableid. \n
*/
REG_OP(TableToResource)
.INPUT(table_id, TensorType({DT_INT32}))
    .OUTPUT(table_handle, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(TableToResource)

/**
* @brief embedding feature_id trans to offset_id. \n

* @par Inputs:
* @li feature_id: A Tensor, dtype is int64.

* @par Outputs:
* @li offset_id: A Tensor with same shape of feature_id, dtype is int32. \n
*/
REG_OP(EmbeddingFeatureMapping)
.INPUT(feature_id, TensorType({DT_INT64}))
    .OUTPUT(offset_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(EmbeddingFeatureMapping)

/**
* @brief embedding hashtable resource applyadagrad. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li lr: A Scalar, dtype is DT_FLOAT/DT_FLOAT16. 0-D. indicates the learning rate.
* @li grad: A Tensor, dtype is DT_FLOAT/DT_FLOAT16. 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable. \n
*/
REG_OP(EmbeddingApplyAdaGrad)
.INPUT(var_handle, TensorType({DT_RESOURCE}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(EmbeddingApplyAdaGrad)

/**
* @brief embedding compute var export. \n

* @par Inputs:
* @li file_path: A String, indicates the export file path.
* @li ps_id: A Int, dtype is int32, indicates the ps server id.
* @li table_id: A Int, dtype is int32, indicates the hashtable id. \n

* @par Attributes:
* @li table_name: A List String. represents table name corresponding to table id . \n
*/
REG_OP(EmbeddingComputeVarExport)
.INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .ATTR(table_name, ListString, {})
    .OP_END_FACTORY_REG(EmbeddingComputeVarExport)

/**
* @brief embedding compute var import. \n

* @par Inputs:
* @li file_path: A String, indicates the import file path.
* @li ps_id: A Int, dtype is int32, indicates the ps server id.
* @li table_id: A Int, dtype is int32, indicates the hashtable id.

* @par Attributes:
* @li table_name: A List String. represents table name corresponding to table id . \n
*/
REG_OP(EmbeddingComputeVarImport)
.INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .ATTR(table_name, ListString, {})
    .OP_END_FACTORY_REG(EmbeddingComputeVarImport)

/**
* @brief Computes the output as scale * (x + bias) if x+bias > 0 and scale * negative_slope * (x+bias)
* if x+bias <= 0 . \n

* @par Inputs:
* Two input:
* x: A Tensor. Must be one of the following types: float32, float16, double.
* bias: A Tensor. Must be one of the following types: float32, float16, double.
*
* @par Attributes:
* negative_slope: A float32. Defaults to "0.2".
* sacle: A float32. Defaults to "2**0.5".
*
* @par Outputs:
* y: A Tensor. Has the same type as "x".
* @par Third-party framework compatibility
* Compatible with the mmcv operator FusedBiasLeakyrelu.
*/
REG_OP(FusedBiasLeakyRelu)
.INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .ATTR(negative_slope, Float, 0.2)
    .ATTR(scale, Float, 1.414213562373)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FusedBiasLeakyRelu)

/**
* @brief Computes the output as scale * gradients if features > 0 and
* negative_slope * gradients * scale if features <= 0 . \n

* @par Inputs:
* Two inputs, including:
* @li y_grad: A Tensor. Must be one of the following types: float16, float32, double.
* @li features: A Tensor. Has the same type as "gradients" . \n

* @par Attributes:
* negative_slope: A float32. Defaults to "0.2" . \n
* scale : A float32. Defaults to "2**0.5"

* @par Outputs:
* x_grad: A Tensor. Has the same type as "y_grad" . \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator FusedBiasLeakyReluGrad.
*/
REG_OP(FusedBiasLeakyReluGrad)
.INPUT(y_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(negative_slope, Float, 0.2)
    .ATTR(scale, Float, 1.414213562373)
    .OUTPUT(x_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FusedBiasLeakyReluGrad)


/**
* @brief Set initial values for memory of sizes list . \n

* @par Attributes:
* @li sizes: sizes of workspaces. \n
* @li dtypes: data types of initial values. \n
* @li values_int: integer values to be set. \n
* @li values_float: float values to be set. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(MemSet)
.REQUIRED_ATTR(sizes, ListInt)
    .ATTR(dtypes, ListType, {})
    .ATTR(values_int, ListInt, {})
    .ATTR(values_float, ListFloat, {})
    .OP_END_FACTORY_REG(MemSet)

/**
* @brief Performs the backpropagation of DeformableRoiPool for training scenarios . \n

* @par Inputs:
* Four inputs, including:
* @li grad_output: A 5HD gradient input of type float32
* @li feature_map: A 5HD Tensor of type float32.
* @li rois: ROI position. A 2D Tensor of float32 with shape (N, 5). "N" indicates the number of ROIs,
* the value "5" indicates the indexes of images where the ROIs are located, "x0", "x1", "y0" and "y1".
* @li offset: An optional 5HD Tensor input, specifying the offset of sampled points . \n

* @par Attributes:
* Four attributes, including:
* @li output_size: A required list of 2 ints, obtained based on the shape of "output" of DeformableRoiPool.
* @li spatial_scale: A optional attribute of type float, specifying the scaling ratio of "feature_map"
* to the original image.
* @li sample_ratio: An optional attribute of type int, specifying the horizontal and vertical sampling
* frequency of each output.
* If this attribute is set to "0", the sampling frequency is equal to the rounded up value of "rois",
* which is a floating point number. Defaults to "0".
* @li gamma: An optional attribute of type float, specfying the scaling factor of offset . \n

* @par Outputs:
* @li grad_fm: Gradient added to input "features". Has the same 5HD shape as input "features".
* @li grad_offset: Gradient added to input "offset". Has the same 4D shape as input "offset".
*/
REG_OP(DeformableRoiPoolGrad)
.INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT}))
    .OUTPUT(grad_offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(spatial_scale, Float, 1.0)
    .ATTR(sampling_ratio, Int, 0)
    .ATTR(gamma, Float, 0.1)
    .OP_END_FACTORY_REG(DeformableRoiPoolGrad)

/**
* @brief find an optimal n for shift-n. \n

* @par Inputs:
* @li x: A Tensor. indicates the output of quantizable layers.
* @li scale_d: A Tensor, one number. indicates the scale of data.
* @li scale_w: A Tensor, must be one number or the same size as dim-C when x is NHWC/NCHW.
*              indicates the scale of weight. \n

* @par Outputs:
* @li n: A Tensor, has the same shape as scale_w. indicates the optimal n. \n
*/
REG_OP(SearchN)
.INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scale_d, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scale_w, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(n, TensorType({DT_INT8}))
    .OP_END_FACTORY_REG(SearchN)

/**
* @brief The operator generates three assist matrixs which will be used in AdaptiveAvgPool. \n

* @par Input:
* input_size: A Tensor of type int64.  \n
* output_size: A Tensor of type int64.  \n

* @par Outputs:
* three inputs, including:
* @li left_matrix: A Tensor of type float32.  \n
* @li right_matrix: A Tensor of type float32.  \n
* @li weight_matrix: A Tensor of type float32.  \n
*/
REG_OP(AdaptiveAvgPoolAssistMatrix)
.INPUT(input_size, TensorType({DT_INT64, DT_INT32}))
    .INPUT(output_size, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(left_matrix, TensorType({DT_FLOAT}))
    .OUTPUT(right_matrix, TensorType({DT_FLOAT}))
    .OUTPUT(weight_matrix, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AdaptiveAvgPoolAssistMatrix)

/**
* @brief The operator generates three assist matrixs which will be used in AdaptiveAvgPool2d. \n

* @par Input:
* input_size: A Tensor of type int64.  \n

* @par Outputs:
* three inputs, including:
* @li left_matrix: A Tensor of type float32.  \n
* @li right_matrix: A Tensor of type float32.  \n
* @li weight_matrix: A Tensor of type float32.  \n

* @par Attributes:
* output_size: A required attribute.  \n
*/
REG_OP(AdaptiveAvgPool2dAssistMatrix)
.INPUT(input_size, TensorType({DT_INT64}))
    .OUTPUT(left_matrix, TensorType({DT_FLOAT}))
    .OUTPUT(right_matrix, TensorType({DT_FLOAT}))
    .OUTPUT(weight_matrix, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(output_size, ListInt)
    .OP_END_FACTORY_REG(AdaptiveAvgPool2dAssistMatrix)

/**
* @brief Compute correct bounding box.

* @par Inputs:
* Three inputs, including:
* @li x: A 5D Tensor of type float16 with shape (N, na, no, H, W), na indicates the number of anchors,
* no indicates the number of outputs per anchor, including [xywh, class_num, conf_score].
* @li grid: A 5D Tensor of type float16 with shape (1, na, 2, H, W) for V3/V5 and (1, 1, 2, H, W) for V7,
* the value "2" indicates offsets of coordinates.
* @li anchor_grid: A 5D Tensor of type float16 with shape (1, na, 2, H, W) for V3/V5 and (1, 1, 2, 1, 1) for V7,
* the value "2" indicates anchors relative to the original image.

* @par Attributes:
* @li stride: A required int32, scale for each box.
* @li yolo_version: A required string, specifying the YOLO version, optional [V3, V5, V7].

* @par Outputs:
* @li y: A 5D Tensor of type float16 with shape (N, na, no, H, W), same as the input x.

* @par attention Constraints:
* @li This operator applies to YOLO V3, V5 and V7 networks.
* @par Third-party framework compatibility
* It is a custom operator.
*/
REG_OP(CorrectBBox)
.INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(grid, TensorType({DT_FLOAT16}))
    .INPUT(anchor_grid, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(stride, Int)
    .REQUIRED_ATTR(yolo_version, String)
    .OP_END_FACTORY_REG(CorrectBBox)

/**
* @brief Obtains the ROI feature matrix from the feature map. It is a customized FasterRcnn operator . \n

* @par Inputs:
* Three inputs, including:
* @li features: A 5HD Tensor of type float32 or float16.
* @li rois: ROI position. A 2D Tensor of float32 or float16 with shape (N, 5). "N" indicates the number of ROIs,
*     the value "5" indicates the indexes of images where the ROIs are located, "x0", "y0", "x1", and "y1".
* @li offset: An optional input of type float32 or float16, offset of height and width defaults to a Tensor of zero . \n

* @par Attributes:
* @li spatial_scale: A required attribute of type float32, specifying the scaling ratio of "features"
*     to the original image.
* @li pooled_height: A required attribute of type int32, specifying the H dimension.
* @li pooled_width: A required attribute of type int32, specifying the W dimension.
* @li sampling_ratio: An optional attribute of type int32, specifying the horizontal and vertical sampling frequency
*     of each output. If this attribute is set to "0",
* the sampling frequency is equal to the rounded up value of "rois", which is a floating point number. Defaults to "0".
* @li gamma: An optional attribute of type float32. Defaults to "0.1" . \n
* @par Outputs:
* output: Outputs the feature sample of each ROI position. The format is 5HD Tensor of type float32 or float16.
  The axis N is the number of input ROIs. Axes H, W, and C are consistent
* with the values of "pooled_height",
* "pooled_width", and "features", respectively.
*/
REG_OP(DeformableRoiPool)
.INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(spatial_scale, Float, 1.0)
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(sampling_ratio, Int, 0)
    .ATTR(gamma, Float, 0.1)
    .OP_END_FACTORY_REG(DeformableRoiPool)

/**
 * @brief Generate the attention map of Point-wise Spatial Attention(PSA) \n

 * @par Inputs:
 * x: A Tensor of BasicType that indicates the global attention map from upstream computing. \n

 * @par Outputs:
 * y: A Tensor of BasicType that indicates the generated pixel-wise global attention map. \n

 * @par Attributes:
 * @li psa_type: An Int value of 1 or 2 that indicates the method used to generate pixel-wise global attention map.
 * @li num: An Int value that indicates the batch_size of input x.
 * @li h_feature: An Int value that indicates the hight of input feature map.
 * @li w_feature: An Int value that indicates the width of input feature map.
 * @li h_mask: An Int value that indicates the hight of the over-completed map.
 * @li w_mask: An Int value that indicates the width of the over-completed map.
 * @li half_h_mask: An Int value that indicates half of the hight of input feature map.
 * @li half_w_mask: An Int value that indicates half of the width of the over-completed map. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator PSAMask.\n
 */
REG_OP(PSAMask)
.INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(psa_type, Int)
    .REQUIRED_ATTR(num, Int)
    .REQUIRED_ATTR(h_feature, Int)
    .REQUIRED_ATTR(w_feature, Int)
    .REQUIRED_ATTR(h_mask, Int)
    .REQUIRED_ATTR(w_mask, Int)
    .REQUIRED_ATTR(half_h_mask, Int)
    .REQUIRED_ATTR(half_w_mask, Int)
    .OP_END_FACTORY_REG(PSAMask)

/**
 * @brief Calculate the gradient of operator PSAMask \n

 * @par Inputs:
 * y_grad: A Tensor of BasicType that indicates the passed gradient. \n

 * @par Outputs:
 * x_grad: A Tensor of BasicType that indicates the calculated gradient. \n

 * @par Attributes:
 * @li psa_type: An Int value of 1 or 2 that indicates the method used to generate pixel-wise global attention map.
 * @li num: An Int value that indicates the batch_size of input x.
 * @li h_feature: An Int value that indicates the hight of input feature map.
 * @li w_feature: An Int value that indicates the width of input feature map.
 * @li h_mask: An Int value that indicates the hight of the over-completed map.
 * @li w_mask: An Int value that indicates the width of the over-completed map.
 * @li half_h_mask: An Int value that indicates half of the hight of input feature map.
 * @li half_w_mask: An Int value that indicates half of the width of the over-completed map. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator PSAMask.\n
 */
REG_OP(PSAMaskGrad)
.INPUT(y_grad, TensorType::BasicType())
    .OUTPUT(x_grad, TensorType::BasicType())
    .REQUIRED_ATTR(psa_type, Int)
    .REQUIRED_ATTR(num, Int)
    .REQUIRED_ATTR(h_feature, Int)
    .REQUIRED_ATTR(w_feature, Int)
    .REQUIRED_ATTR(h_mask, Int)
    .REQUIRED_ATTR(w_mask, Int)
    .REQUIRED_ATTR(half_h_mask, Int)
    .REQUIRED_ATTR(half_w_mask, Int)
    .OP_END_FACTORY_REG(PSAMaskGrad)

/**
* @brief Find nearby points in spherical space or spherical layer. \n

* @par Inputs:
* Two inputs, including:
* @li xyz: A 3D Tensor of type float16 or float32, xyz coordinates of the features.
* @li center_xyz: A 3D Tensor of type float16 or float32. centers coordinates of the ball query. \n

* @par Attributes:
* @li min_radius: A required float, minimum radius of the balls.
* @li max_radius: A required float, maximum radius of the balls.
* @li sample_num: A required int, maximum number of features in the balls. \n

* @par Outputs:
* One outputs:
* @li idx: A 3D(B, M, sample_num) Tensor of type int32 with the indices of the features that form the query balls. \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator BallQuery(BallQuery branch).
*/
REG_OP(BallQuery)
.INPUT(xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(center_xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(min_radius, Float)
    .REQUIRED_ATTR(max_radius, Float)
    .REQUIRED_ATTR(sample_num, Int)
    .OP_END_FACTORY_REG(BallQuery)

/**
* @brief Find nearby points in spherical space. \n

* @par Inputs:
* Four inputs, including:
* @li xyz: A 2D Tensor of type float16 or float32, xyz coordinates of the features.
* @li center_xyz: A 2D Tensor of type float16 or float32. Centers coordinates of the ball query.
* @li xyz_batch_cnt: A 1D Tensor of type int32 or int64, Stacked input xyz coordinates nums in
     each batch, just like (N1, N2, ...).
* @li center_xyz_batch_cnt: A 1D Tensor of type int32 or int64. Stacked input centers coordinates nums in
     each batch, just like (M1, M2, ...). \n

* @par Attributes:
* @li max_radius: A required float, maximum radius of the balls.
* @li sample_num: A required int, maximum number of features in the balls. \n

* @par Outputs:
* One outputs:
* @li idx: A 2D(M, sample_num) Tensor of type int32 with the indices of the features that form the query balls. \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator BallQuery(StackBallQuery branch).
*/
REG_OP(StackBallQuery)
.INPUT(xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(center_xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(xyz_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .INPUT(center_xyz_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(max_radius, Float)
    .REQUIRED_ATTR(sample_num, Int)
    .OP_END_FACTORY_REG(StackBallQuery)

/**
 * @brief Find and get the corresponding value from the corresponding ps according to the keys
 * @par Inputs:
 * @li keys: A tensor. Must be int64 type.
 * @li table_id: A tensor. Must be int32 type.
 * @par Outputs:
 * @li values: A Tensor. Must be float32 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li insert_option: Indicates whether lookup supports new value. Defaults to "0".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: Apply memory usage for output or infer shape.
 */
REG_OP(HcomRemoteLookup)
.INPUT(keys, TensorType({DT_INT64}))
    .INPUT(table_id, Int)
    .OUTPUT(values, TensorType({DT_FP32}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(insert_option, Int, 0)
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(HcomRemoteLookup)

/**
 * @brief Workers all find and get the corresponding value from the corresponding ps according to the keys
 * @par Inputs:
 * @li table_id: A tensor. Must be int32 type.
 * @li keys: A tensor. Must be int64 type.
 * @par Outputs:
 * @li values: A Tensor. Must be float32 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li insert_option: Indicates whether lookup supports new value. Defaults to "0".
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: A required integer identifying Apply memory usage for output or infer shape.
 * @li flags: An integer identifying counter filter feature.
 */
REG_OP(HcomCollRemoteLookup)
.INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FP32}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(insert_option, Int, 0)
    .ATTR(group, String, "hccl_world_group")
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(flags, Int, 0)
    .OP_END_FACTORY_REG(HcomCollRemoteLookup)

/**
 * @brief Workers send the keys and values to ps according to keys
 * @par Inputs:
 * @li table_id: A tensor. Must be int32 type.
 * @li keys: A tensor. Must be int64 type.
 * @li values: A Tensor. Must be float32 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: Apply memory usage for output or infer shape.
 */
REG_OP(HcomCollRemoteUpdate)
.INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FP32}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(group, String, "hccl_world_group")
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(HcomCollRemoteUpdate)

/**
 * @brief Workers all find and get the corresponding value from the corresponding ps according to the keys. Used with
 * HcomCollRemoteUpdatePaired.
 * @par Inputs:
 * @li table_id: A tensor. Must be int32 type.
 * @li keys: A tensor. Must be int64 type.
 * @par Outputs:
 * @li values: A Tensor. Must be float32 type.
 * @li indices: A Tensor. Recovery matrix. Must be int64 type.
 * @li num_uniqued: A Tensor. Number of Recovery matrix. Must be int64 type.
 * @li ps_segments: A Tensor. Offset and size of buffer for pss. Must be int64 type.
 * @li ps_segments_num: A Tensor. Number of ps_segments. Must be int64 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li insert_option: Indicates whether lookup supports new value. Defaults to "0".
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: A required integer identifying Apply memory usage for output or infer shape.
 * @li flags: An integer identifying counter filter feature.
 */
REG_OP(HcomCollRemoteLookupPaired)
.INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FP32}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(num_uniqued, TensorType({DT_INT64}))
    .OUTPUT(ps_segments, TesnorType({DT_INT64}))
    .OUTPUT(ps_segments_num, TesnorType({DT_INT64}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(insert_option, Int, 0)
    .ATTR(group, String, "hccl_world_group")
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(flags, Int, 0)
    .OP_END_FACTORY_REG(HcomCollRemoteLookupPaired)

/**
 * @brief Workers send the keys and values to ps according to keys. Used with HcomCollRemoteLookupPaired.
 * @par Inputs:
 * @li table_id: A tensor. Must be int32 type.
 * @li keys: A tensor. Must be int64 type.
 * @li values: A Tensor. Must be float32 type.
 * @li indices: A Tensor. Recovery matrix. Must be int64 type.
 * @li num_uniqued: A Tensor. Number of Recovery matrix. Must be int64 type.
 * @li ps_segments: A Tensor. Offset and size of buffer for pss. Must be int64 type.
 * @li ps_segments_num: A Tensor. Number of ps_segments. Must be int64 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: Apply memory usage for output or infer shape.
 */
REG_OP(HcomCollRemoteUpdatePaired)
.INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FP32}))
    .INPUT(indices, TesnorType({DT_INT64}))
    .INPUT(num_uniqued, TesnorType({DT_INT64}))
    .INPUT(ps_segments, TesnorType({DT_INT64}))
    .INPUT(ps_segments_num, TesnorType({DT_INT64}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(group, String, "hccl_world_group")
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(HcomCollRemoteUpdatePaired)

/**
 * @brief Calculate that aggregates input data
 * @par Inputs:
 * @li x: A tensor of type float32, int32, int8, int16, float16, int64, uint64
 * @par Outputs:
 * @li y: A tensor of type float32, int32, int8, int16, float16, int64, uint64.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator root_rank.
 * @li group: A string identifying the group name of ranks participating in
  the op.
 * @li rank_size: A required integer identifying the rank size.
 */
REG_OP(HcomGather)
.INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64}))
    .REQUIRED_ATTR(root_rank, Int)
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(rank_size, Int)
    .OP_END_FACTORY_REG(HcomGather)

/**
* @brief Find a min polygon from the point set in the operator MinAreaPolygons. \n

* @par Inputs:
* @li pointsets: A 2D Tensor with shape (N, 18), format ND, dtype must be one
 of the following types: float16, float32, double. \n

* @par Outputs:
* @li polygons: A 2D Tensor with shape (N, 8), format ND, dtype must be one of
 the following types: float16, float32, double.  \n
*/
REG_OP(MinAreaPolygons)
.INPUT(pointsets, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(polygons, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(MinAreaPolygons)

/**
* @brief Determine if the target points set is inside polygons. \n

* @par Inputs:
* @li points: A 2D Tensor with shape (N, 2), format ND, dtype must be float32. \n
* @li polygons: A 2D Tensor with shape (M, 8), format ND, dtype must be float32.
*     This parameter will be transposed to be (8, M) before passed to the operator. \n

* @par Outputs:
* @li output: A 2D Tensor with shape (N, M), format ND, dtype must be float32.  \n
*/
REG_OP(PointsInPolygons)
.INPUT(points, TensorType({DT_FLOAT}))
    .INPUT(polygons, TensorType({DT_FLOAT}))
    .OUTPUT(output, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(PointsInPolygons)

/**
* @brief Calculate the index and distance of the nearest three point to the target point.
* @par Inputs:
* Two input:
* xyz1: The set of target points.
* xyz2: The set of compare points. \n

* @par Outputs:
* dist: A Tensor, the distance of the nearest point to the target point.
* idx: A Tensor, the index of the nearest point to the target point. \n
*/
REG_OP(ThreeNN)
.INPUT(xyz1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(xyz2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(dist, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(ThreeNN)

/**
 * @brief Calculate the voxels of cloud points \n

 * @par Inputs:
 * Three inputs, including:
 * @li points: the shape is [M, C], points[:3] contain xyz points and points[3:] contain other information.
 * @li voxel_size: the size of voxel with the shape of [3].
 * @li coors_range:the coordinate range of voxel with the shape of [6]. \n

 * @par Outputs:
 * Four outputs, including:
 * @li voxels: the output voxels with the shape of [M, max_points, C].
 * @li coors: the voxel coordinates with shape of [M, 3].
 * @li num_points_per_voxel: the number of points per voxel with the shape of [M].
 * @li voxel_num: the number of voxels. \n

 * @par Attributes:
 * Three attrs, including:
 * @li max_points: maximum points contained in a voxel.
 * @li max_voxels: maximum voxels this op create.
 * @li deterministic: An optional attr, only support true now, false is faster. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator Voxelization.\n
 */
REG_OP(Voxelization)
.INPUT(points, TensorType({DT_DOUBLE,DT_FLOAT,DT_FLOAT16}))
    .INPUT(voxel_size, TensorType({DT_DOUBLE,DT_FLOAT,DT_FLOAT16}))
    .INPUT(coors_range, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(voxels, TensorType({DT_DOUBLE,DT_FLOAT,DT_FLOAT16}))
    .OUTPUT(coors, TensorType({DT_INT32}))
    .OUTPUT(num_points_per_voxel, TensorType({DT_INT32}))
    .OUTPUT(voxel_num, TensorType({DT_INT32}))
    .ATTR(max_points, Int, 35)
    .ATTR(max_voxels, Int, 20000)
    .ATTR(deterministic, Bool, true)
    .OP_END_FACTORY_REG(Voxelization)

/**
 * @brief Encoding the orientation information and generating orientation-sensitive features. \n

 * @par Inputs:
 * Two inputs, including:
 * @li x: Input features with shape [num_output_planes, num_input_planes, num_orientations, H, W].
 * @li indices: Indices with shape [num_orientations, H, W, num_rotations]. \n

 * @par Outputs:
 * One output, including:
 * @li y: Refined features with shape [num_output_planes * num_rotations, num_input_planes * num_orientations, H, W]. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator ActiveRotatedFilter.\n
 */
REG_OP(ActiveRotatedFilter)
.INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OP_END_FACTORY_REG(ActiveRotatedFilter)

/**
 * @brief The backward of ActiveRotatedFilter. \n

 * @par Inputs:
 * Two inputs, including:
 * @li y_grad: Input features with shape [num_output_planes * num_rotations, num_input_planes * num_orientations, H, W].
 * @li indices: Indices with shape [num_orientations, H, W, num_rotations]. \n

 * @par Outputs:
 * One output, including:
 * @li x_grad: Refined features with shape [num_output_planes, num_input_planes, num_orientations, H, W]. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator ActiveRotatedFilterGrad.\n
 */
REG_OP(ActiveRotatedFilterGrad)
.INPUT(y_grad, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OP_END_FACTORY_REG(ActiveRotatedFilterGrad)

/**
* @brief Blend face iamge to the backgroud.
*
* @par Inputs:
* @li face_img: A 3D Tensor, dtype is uint8 or float32, shape is (h, w, 3). The input face image.
* @li face_rect: A 1D Tensor, dtype is int32, shape is (4,). The coordinates of the face image in the backgroud.
* @li face_mask: A 3D Tensor, dtype is float32, shape is (h, w, 1).
* @li acc_face: A 3D Tensor, dtype is float32, shape is (H, W, 3).
* @li acc_mask: A 3D Tensor, dtype is float32, shape is (H, W, 3).
* @li max_mask: A 3D Tensor, dtype is float32, shape is (H, W, 3).
*
* @par Outputs:
* @li acc_face: A 3D Tensor, Has the same type and shape as input "acc_face".
* @li acc_mask: A 3D Tensor, Has the same type and shape as input "acc_mask".
* @li max_mask: A 3D Tensor, Has the same type and shape as input "max_mask". \n
*/
REG_OP(BlendFaceBgPartOne)
.INPUT(face_img, TensorType({DT_UINT8, DT_FLOAT}))
    .INPUT(face_rect, TensorType({DT_INT32}))
    .INPUT(face_mask, TensorType({DT_FLOAT}))
    .INPUT(acc_face, TensorType({DT_FLOAT}))
    .INPUT(acc_mask, TensorType({DT_FLOAT}))
    .INPUT(max_mask, TensorType({DT_FLOAT}))
    .OUTPUT(acc_face, TensorType({DT_FLOAT}))
    .OUTPUT(acc_mask, TensorType({DT_FLOAT}))
    .OUTPUT(max_mask, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BlendFaceBgPartOne)

/**
* @brief Blend face iamge to the backgroud Part Two.
*
* @par Inputs:
* @li acc_face: A 3D Tensor, dtype is float32, shape is (H, W, 3).
* @li acc_mask: A 3D Tensor, dtype is float32, shape is (H, W, 3).
* @li max_mask: A 3D Tensor, dtype is float32, shape is (H, W, 3).
* @li bg_img: A 3D Tensor, dtype is float32 or uint8, shape is (H, W, 3), the input background image.
*
* @par Attributes:
* @li epsilon: A scalar of the same type as "var".
*
* @par Outputs:
* @li fused_img: A 3D Tensor, Has the same type and shape as input "acc_face". \n
*/
REG_OP(BlendFaceBgPartTwo)
.INPUT(acc_face, TensorType({DT_FLOAT}))
    .INPUT(acc_mask, TensorType({DT_FLOAT}))
    .INPUT(max_mask, TensorType({DT_FLOAT}))
    .INPUT(bg_img, TensorType({DT_UINT8, DT_FLOAT}))
    .OUTPUT(fused_img, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-12)
    .OP_END_FACTORY_REG(BlendFaceBgPartTwo)

/**
* @brief Convert the image from YUV to Raw.
*
* @par Inputs:
* @li img_channel_0: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 0.
* @li img_channel_1: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 1.
* @li img_channel_2: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 2.
* @li img_channel_3: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 3.
* @li img_size: A 1D Tensor, dtype is int32, shape is (2,).
*     The data is h_out and w_out, which indicates the output height and width.
* @li gamma: A 1D Tensor, dtype is float32, shape is (4,).
*
* @par Attributes:
* @li bayer_pattern: A string. choce calculate mode, the value must be one of ["binning", "quad"]. Default: "binning".
*
* @par Outputs:
* @li raw_img: A 2D Tensor, dtype is uint16, shape is (h_out, w_out). the output raw image. \n
*/
REG_OP(ImgRawDecodePostHandle)
.INPUT(img_channel_0, TensorType({DT_UINT16}))
    .INPUT(img_channel_1, TensorType({DT_UINT16}))
    .INPUT(img_channel_2, TensorType({DT_UINT16}))
    .INPUT(img_channel_3, TensorType({DT_UINT16}))
    .INPUT(img_size, TensorType({DT_INT32}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .OUTPUT(raw_img, TensorType({DT_UINT16}))
    .ATTR(bayer_pattern, String, "binning")
    .OP_END_FACTORY_REG(ImgRawDecodePostHandle)

/**
* @brief Convert the image from YUV to Raw.
*
* @par Inputs:
* @li img_channel_0: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 0.
* @li img_channel_1: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 1.
* @li img_channel_2: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 2.
* @li img_channel_3: A 2D Tensor, dtype is uint16, shape is (h, w). The input image of channel 3.
* @li gamma: A 1D Tensor, dtype is float32, shape is (4,).
* @li bayer_coordinate: A 1D Tensor, dtype is int32, shape is (4,).
*     The data is the left top and right bottom coordinates, [lt_x, lt_y, rb_x, rb_y].
* @li bayer_params: A 1D Tensor, dtype is float32, shape is (8,).
*     The data is bayer params, [r_gain, g_gain, b_gain, iso, ev_gain, iso_long, evSL, exposure_gain].
* @li bayer_ptn: A 1D Tensor, dtype is int32, shape is (4,).
*
* @par Outputs:
* @li raw_img: A 2D Tensor, dtype is float32, shape is (h_out, w_out). the output raw image. \n
*/
REG_OP(ImgRawDecodePostHandleV2)
.INPUT(img_channel_0, TensorType({DT_UINT16}))
    .INPUT(img_channel_1, TensorType({DT_UINT16}))
    .INPUT(img_channel_2, TensorType({DT_UINT16}))
    .INPUT(img_channel_3, TensorType({DT_UINT16}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .INPUT(bayer_coordinate, TensorType({DT_INT32}))
    .INPUT(bayer_params, TensorType({DT_FLOAT}))
    .INPUT(bayer_ptn, TensorType({DT_INT32}))
    .OUTPUT(raw_img, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ImgRawDecodePostHandleV2)

/**
* @brief YUV4442YUV422. Convert the image from yuv444 to yuv422. \n

* @par Inputs:
* @li x: A 3D Tensor, dtype is float16, shape is (h, w, 4). The input yuv444 data. \n
*
* @par Outputs:
* @li y: A 3D Tensor, dtype is uint8, shape is (h, w, 2). The output yuv422 data. \n
*/
REG_OP(YUV4442YUV422)
.INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(YUV4442YUV422)

/**
* @brief RGB2YUV422. Convert the image from rgb to yuv422. \n

* @par Inputs:
* rgb: A Tensor of type uint8. \n
* @par Outputs:
* yuv: A Tensor of type uint8. \n

* @attention Constraints:
* Input images is a tensor of at least 3 dimensions. The last dimension is
* interpretted as channels, and must be three . \n
*/
REG_OP(RGB2YUV422)
.INPUT(rgb, TensorType({DT_UINT8}))
    .OUTPUT(yuv, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(RGB2YUV422)

/**
* @brief Function FlashAttentionScore. \n

* @par Inputs:
* six inputs, including:
* @li query: A matrix Tensor. The type support float16, bf16, float32 .
* @li key: A matrix Tensor. The type support float16, bf16, float32.
* @li value: A matrix Tensor. The type support float16, bf16, float32.
* @li real_shift: A matrix Tensor. The type support float16, bf16, float32.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float16, bf16, float32.
* @li atten_mask: A matrix Tensor. The type support float16, bf16, float32.

* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens.
* @li next_tockens: A int. Next tokens.
* @li head_num: A int. The number of the heads.
* @li is_flash: A bool. If True, use flash attention algorithm.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH"]. Default: "BSH".
*
* @par Outputs:
* softmax_max: A matrix Tensor. The type support float32.
* softmax_sum: A matrix Tensor. The type support float32.
* softmax_out: A matrix Tensor. The type support float16, bf16, float32.
* attention_out: A matrix Tensor. The type support float16, bf16, float32.


* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(FlashAttentionScore)
.INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(real_shift, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT1, DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(is_flash, Bool)
    .REQUIRED_ATTR(input_layout, String)
    .OP_END_FACTORY_REG(FlashAttentionScore)

/**
* @brief Function FlashAttentionScoreGrad. \n

* @par Inputs:
* twelve inputs, including:
* @li query: A matrix Tensor. The type support float16, bf16, float32.
* @li key: A matrix Tensor. The type support float16, bf16, float32.
* @li value: A matrix Tensor. The type support float16, bf16, float32.
* @li dy: A matrix Tensor. The type support float16, bf16, float32.
* @li real_shift: A scalar. The type support float16, bf16, float32.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float16, bf16, float32.
* @li atten_mask: A matrix Tensor. The type support float16, bf16, float32.
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
* @li softmax_in: A matrix Tensor. The type support float16, bf16, float32.
* @li attention_in: A matrix Tensor. The type support float16, bf16, float32.


* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens.
* @li next_tockens: A int. Next tokens.
* @li head_num: A int. The number of the heads.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH"]. Default: "BSH".


* @par Outputs:
* dq: A matrix Tensor. The type support float16, bf16, float32.
* dk: A matrix Tensor. The type support float16, bf16, float32.
* dv: A matrix Tensor. The type support float16, bf16, float32.
* dpse: A matrix Tensor. The type support float16, bf16, float32.


* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(FlashAttentionScoreGrad)
.INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_in, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(attention_in, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(dq, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(dk, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(dv, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(dpse, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 65536)
    .ATTR(next_tockens, Int, 65536)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .OP_END_FACTORY_REG(FlashAttentionScoreGrad)

/**
* @brief Multiplies sparse updates into a variable reference . \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor List.
* Must be one of the following types: float16, float, int16, int32, int64, int8, uint16, uint32, uint64, uint8, bfloat16
* @li indices: An ND Tensor.
* Must be one of the following types: int32 or int64
* @li updates: An ND Tensor .
* Must be one of the following types: float16, float, int16, int32, int64, int8, uint16, uint32, uint64, uint8, bfloat16
* @li mask: An ND Tensor .
* Must be one of the following types: uint8
* @par Attributes:
* @li reduction: An optional attribute. Defaults to string "update"
* @li axis: An optional attribute. Defaults to -2.

* @par Outputs:
* var: An ND Tensor List. Has the same type and format as input "var" . \n

* @par Third-party framework compatibility
* Compatible with the Mindspore operator Scatter.
*/
REG_OP(ScatterList)
    .DYNAMIC_INPUT(var, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                    DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(indice, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                DT_UINT16, DT_UINT32, DT_UINT64}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .DYNAMIC_OUTPUT(var, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                     DT_UINT16, DT_UINT32, DT_UINT64}))
    .ATTR(reduce, String, "update")
    .ATTR(axis, Int, -2)
    .OP_END_FACTORY_REG(ScatterList)

/**
* @brief Function MultiHeadAttentionScoreGrad. \n

* @par Inputs:
* twelve inputs, including:
* @li query: A matrix Tensor. The type support float32.
* @li key: A matrix Tensor. The type support float32.
* @li value: A matrix Tensor. The type support float32.
* @li dy: A matrix Tensor. The type support float32.
* @li real_shift: A scalar. The type support float32.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float32.
* @li atten_mask: A matrix Tensor. The type support float32.
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
* @li softmax_in: A matrix Tensor. The type support float32.
* @li attention_in: A matrix Tensor. The type support float32.


* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens.
* @li next_tockens: A int. Next tokens.
* @li head_num: A int. The number of the heads.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH"]. Default: "BSH".


* @par Outputs:
* dq: A matrix Tensor. The type support float32.
* dk: A matrix Tensor. The type support float32.
* dv: A matrix Tensor. The type support float32.
* dpse: A matrix Tensor. The type support float32.


* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(MultiHeadAttentionScoreGrad)
.INPUT(query, TensorType({DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT32}))
    .INPUT(dy, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_in, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(attention_in, TensorType({DT_FLOAT32}))
    .OUTPUT(dq, TensorType({DT_FLOAT32}))
    .OUTPUT(dk, TensorType({DT_FLOAT32}))
    .OUTPUT(dv, TensorType({DT_FLOAT32}))
    .OUTPUT(dpse, TensorType({DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 65536)
    .ATTR(next_tockens, Int, 65536)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .OP_END_FACTORY_REG(MultiHeadAttentionScoreGrad)

REG_OP(IncreFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .DYNAMIC_INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .DYNAMIC_INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_FLOAT32}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(IncreFlashAttention)

REG_OP(MlaProlog)
    .INPUT(token_x, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_dq, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uq_qr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uk, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight_dkv_kr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(rmsnorm_gamma_cq, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rmsnorm_gamma_ckv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(cache_index, TensorType({DT_INT64}))
    .INPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(dequant_scale_x, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dq, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_uq_qr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dkv_kr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckv, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scales_cq, TensorType({DT_FLOAT}))
    .OUTPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(rmsnorm_epsilon_cq, Float, 1e-05)
    .ATTR(rmsnorm_epsilon_ckv, Float, 1e-05)
    .ATTR(cache_mode, String, "PA_BSND")
    .OP_END_FACTORY_REG(MlaProlog)

REG_OP(MlaPrologV2)
    .INPUT(token_x, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_dq, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uq_qr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uk, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight_dkv_kr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(rmsnorm_gamma_cq, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rmsnorm_gamma_ckv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(cache_index, TensorType({DT_INT64}))
    .INPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(dequant_scale_x, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dq, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_uq_qr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dkv_kr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckv, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scales_cq, TensorType({DT_FLOAT}))
    .OUTPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(dequant_scale_q_nope, TensorType({DT_FLOAT}))
    .ATTR(rmsnorm_epsilon_cq, Float, 1e-05)
    .ATTR(rmsnorm_epsilon_ckv, Float, 1e-05)
    .ATTR(cache_mode, String, "PA_BSND")
    .OP_END_FACTORY_REG(MlaPrologV2)

REG_OP(MlaPrologV3) 
    .INPUT(token_x, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_dq, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uq_qr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uk, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight_dkv_kr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(rmsnorm_gamma_cq, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rmsnorm_gamma_ckv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(cache_index, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale_x, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dq, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_uq_qr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dkv_kr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckv, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scales_cq, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(actual_seq_len, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(k_nope_clip_alpha, TensorType({DT_FLOAT}))
    .OUTPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(dequant_scale_q_nope, TensorType({DT_FLOAT}))
    .OUTPUT(query_norm, TensorType({DT_INT8, DT_BF16}))
    .OUTPUT(dequant_scale_q_norm, TensorType({DT_FLOAT}))
    .ATTR(rmsnorm_epsilon_cq, Float, 1e-05)
    .ATTR(rmsnorm_epsilon_ckv, Float, 1e-05)
    .ATTR(cache_mode, String, "PA_BSND")
    .ATTR(query_norm_flag, Bool, false)
    .ATTR(weight_quant_mode, Int, 0)
    .ATTR(kv_cache_quant_mode, Int, 0)
    .ATTR(query_quant_mode, Int, 0)
    .ATTR(ckvkr_repo_mode, Int, 0)
    .ATTR(quant_scale_repo_mode, Int, 0)
    .ATTR(tile_size, Int, 128)
    .ATTR(qc_qr_scale, Float, 1.0)
    .ATTR(kc_scale, Float, 1.0)
    .OP_END_FACTORY_REG(MlaPrologV3)

REG_OP(PromptFlashAttention)
.INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(pre_tokens, Int, 214748647)
    .ATTR(next_tokens, Int, 0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 1)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(PromptFlashAttention)
/**
* @brief paste sub img.
*
* @par Inputs:
* @li patch_img: A 3D Tensor, dtype is uint8 or float16 or float32, shape is (h, w, c). The input image.
* @li patch_coord: A 1D Tensor, dtype is int32, shape is (4,). The coordinates in the combined img.
* @li core_area_coord: A 1D Tensor, dtype is int32, shape is (4,). The coordinates in the patch img
* @li combine_img: A 3D Tensor, dtype is uint8 or float16 or float32, shape is (H, W, C).
*
* @par Outputs:
* @li combine_img: A 3D Tensor, Has the same type and shape as input "combine_img".
*
* @par Attr
* @li scale: Float, scale of coordinates.\n
*/
REG_OP(PasteSubImg)
.INPUT(patch_img, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT32}))
    .INPUT(patch_coord, TensorType({DT_INT32}))
    .INPUT(core_area_coord, TensorType({DT_INT32}))
    .INPUT(combine_img, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(combine_img, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT32}))
    .REQUIRED_ATTR(scale, Float)
    .OP_END_FACTORY_REG(PasteSubImg)

/**
* @brief RotatedFeatureAlignGrad:Calculate the gradient of input features according to
* the gradient of output features. \n

* @par Inputs:
* @li dy: A tensor of type float32. The gradient of output features.
* @li bboxes: A tensor of type float32. The position information of bboxes. \n

* @par Outputs:
* @li dx: A tensor of type float32. The gradient of input features. \n

* @par Attributes:
* @li spatial_scale: A required float32. The scale of feature map to initial image.
* @li points: An optional int. Defaults to "1". The number of sample points. \n

* @par Third-party framework compatibility
* Compatible with MMCV RotatedFeatureAlign operator.
*/

REG_OP(RotatedFeatureAlignGrad)
.INPUT(dy, TensorType({DT_FLOAT}))
    .INPUT(bboxes, TensorType({DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(spatial_scale, Float)
    .ATTR(points, Int, 1)
    .OP_END_FACTORY_REG(RotatedFeatureAlignGrad)

/**
* @brief Computes the transpose of convolution 2d with respect to the input.
* @par Inputs:
* Five inputs:
* @li x: A Tensor of type int8.
* The format is NHWC or NCHW.
* @li filter_compress: A Tensor of type int8. Must have the same type as "x".
* The format is NHWC or NCHW or HWCN.
* @li compress_index: A Tensor of type int8. Index for decompression.
* Must have the same type and format as "filter_compress".
* @li bias: An optional 1D tensor of the same type as "y".
* @li offset_w: An optional 1D tensor for quantized inference. Type is int8.
* @par Required Attributes:
* @li input_size: An integer vector representing the shape of input.
* @li strides: A tuple/list of 4 integers.
* Specifies the stride of the sliding window for each dimension of "x".
* The N and C dimensions must be 1. Has the same format as "x".
* @li pads: A required list or tuple of int32. Padding added to each dimension
* of the input.
* @par Attributes:
* Six attributes:
* @li dilations: A tuple/list of 4 integers. The dilation factor for each dimension
* of input. The N and C dimensions must be 1. Has the same format as "x".
* @li groups: Number of blocked connections from input channels to output channels.
* Defaults to "1".
* @li data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
* Specify the data format of the input and output data.
* @li output_padding: The size will be added in the output shape. Defaults
* to [0, 0, 0, 0].
* @li offset_x: An optional int. Input offset, used for quantized inference.
* Defaults to "0".
* @li alg: An optional string from "weiight_unzip", "weight_sparse_4_2"
* @par Outputs:
* y: A Tensor of type int32.
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Conv2DTransposeDCompress)
.INPUT(x, TensorType({DT_INT8}))
    .INPUT(filter_compress, TensorType({DT_INT8}))
    .INPUT(compress_index, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(input_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .ATTR(alg, String, "weight_sparse_4_2")
    .OP_END_FACTORY_REG(Conv2DTransposeDCompress)

/**
* @brief Detect whether there is Inf or Nan in scaled_grads, set found_inf to 1 if it exists,
* and do not operate on found_inf if it does not. Finally, multiply all values of scaled_grads by inv_scale
* @par Inputs:
 * Three inputs:
 * @li scaled_grads: A tensor list containing multiple tensors, can be float16, float,
 * meanwhile, this value is also an output, store the value multiplied by inv_scale.
 * @li found_inf: A tensor with only one element, the shape must be (1,), must be float,
 * meanwhile, this value is also an output, indicating whether there is Inf or Nan present.
 * @li inv_scale: A tensor with only one element, the shape must be (1,), must be float.
*/
REG_OP(ForeachNonFiniteCheckAndUnscale)
.DYNAMIC_INPUT(scaled_grads, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(found_inf, TensorType({DT_FLOAT}))
    .INPUT(inv_scale, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ForeachNonFiniteCheckAndUnscale)

/**
* @brief multiply scalar foreach element in each tensor in tesnorlist, this is an in-place operation.
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors, can be float16, float, and int32.
 * @li scalar: A scalar to be multiplied, the data type must be the same as tensors.
*/
REG_OP(ForeachMulScalarInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMulScalarInplace)

/**
* @brief Writes the input data of the corresponding subscript to the specified register.
* @par Inputs:
* One inputs:
* @li x: A Tensor of type uint64.
* The format is ND.
* @par Attributes:
* One attribute:
* @li indices: Int, indication a specific subscript value.
*/
REG_OP(SwitchByIndex)
.INPUT(x, TensorType({DT_UINT64}))
    .REQUIRED_ATTR(indices, Int)
    .OP_END_FACTORY_REG(SwitchByIndex)


/**
* @brief Computes the transpose of convolution 2d with respect to the input.
* @par Inputs:
* Four inputs:
* @li x1: A Tensor of type int8. The format is ND.
* @li X2: A Tensor of type int8. Must have the same type as "x". The format is ND.
* @li bias: A Tensor of type int32. The format is ND.
* @li deq_scale: A tensor for quantized inference. The format is NHWC. Type is uint64.
* @par Required Attributes:
* y: A Tensor of type fp16. The format is ND.
* @par Attributes:
* Two attributes:
* @li adj_x1: A bool, if true means x1 is transposed.
* @li adj_x2: A bool, if true means x2 is transposed.
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(QuantBatchMatmul)
.INPUT(x1, TensorType({DT_INT8}))
    .INPUT(x2, TensorType({DT_INT8}))
    .INPUT(bias, TensorType({DT_INT32}))
    .INPUT(deq_scale, TensorType({DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(QuantBatchMatmul)


REG_OP(MoeFFN)
.INPUT(x, TensorType({DT_INT8, DT_FLOAT16}))
    .INPUT(expert_tokens, TensorType({DT_INT64}))
    .INPUT(weight1, TensorType({DT_INT8, DT_FLOAT16}))
    .OPTIONAL_INPUT(bias1, TensorType({DT_INT32, DT_FLOAT16}))
    .OPTIONAL_INPUT(weight2, TensorType({DT_INT8, DT_FLOAT16}))
    .OPTIONAL_INPUT(bias2, TensorType({DT_INT32, DT_FLOAT16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT16}))
    .ATTR(activation, String, "gelu")
    .OP_END_FACTORY_REG(MoeFFN)


/**
* @brief compute init routing for moe input.
* @par Inputs:
* @li x: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li row_idx: A Tensor. Type is:Int32.
* @li expert_idx: A Tensor. Type is:Int32.
* @par Outputs:
* @li expanded_x: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li expanded_row_idx: A Tensor. Type is:Int32.
* @li expanded_expert_idx: A Tensor. Type is:Int32.
* @par Attributes:
* @li active_num: Required parameter. Type is:Int32.
*/
REG_OP(MoeInitRouting)
    .INPUT(x, "T1")
    .INPUT(row_idx, "T2")
    .INPUT(expert_idx, "T2")
    .OUTPUT(expanded_x, "T1")
    .OUTPUT(expanded_row_idx, "T2")
    .OUTPUT(expanded_expert_idx, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .REQUIRED_ATTR(active_num, Int)
    .OP_END_FACTORY_REG(MoeInitRouting)

    
/**
 * @brief compute init routing for moe.
 * @par Inputs:
 * @li x: A 2D Tensor. Shape is: (B*S, H). Type is:Int8, BFloat16, Float16 or Float32. Format support ND.
 * @li expert_idx: A 2D Tensor. Shape is: (B*S, K). Type is:Int32. Format support ND.
 * @li scale: A 1D or 2D Tensor. Shape is: (B*S) or (B*S, H). Type is:Float32. Format support ND.
 * @li offset: A 2D Tensor. Shape is: (expert_end - expert_start, 1) or (expert_end - expert_start, H).
               Type is:Float32. Format support ND.
 * @par Outputs:
 * @li expanded_x: A 2D Tensor. Shape is: (B*S*K, H). Type is: Int8, BFloat16, Float16 or Float32. 
                   The data type must be the same as that of x. Format support ND.
 * @li expanded_row_idx: A 1D Tensor. Shape is: (B*S*K). Type is: Int32. Format support ND.
 * @li expert_tokens_count_or_cumsum: A 1D Tensor. represents the number of tokens processed by each expert and the
                                      cumulative value. The value is controlled by expert_tokens_num_flag to output.
                                      Type is:Int64. shape is (expert_end - expert_start, ). Format support ND.
 * @li expanded_scale: A 1D Tensor. Shape is: (B*S*K). Type is: Float32. 
                       The data type must be the same as that of scale. Format support ND.
 * @par Attributes:
 * @li active_num: Optional parameter. Type is:Int32. identify activate scenario. The value 0 indicates a non-active
 *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size
 *                 of axis 0 of grad_expanded_x must be equal to the value of active_num. Default: 0.
 * @li expert_capacity: Optional parameter. Type is:Int32. The max tokens count of every expert. Default: 0.
 * @li expert_num: Optional parameter. Type is:Int32. Default: 0.
 * @li drop_pad_mode: Optional parameter. Type is:Int32. The value is 0(dropless) or 1(dropPad). Default: 0.
 * @li expert_tokens_num_type: Optional parameter. Type is:Int32. The value is 0(compute tokens cumsum) or
                               1(compute tokens count), which in dropPad scenario. Default: false.
 * @li expert_tokens_num_flag: Optional parameter. Type is:Bool. The value is true (compute tokens) or
                               false(do not compute tokens), which in dropPad scenario. Default: false.
 * @li quant_mode: Optional parameter. Type is:Int. The value is in [-1(unquant), 0(static), 1(dynamic), 2(fp8_e5m2), 3(fp8_e4m3fn)] of quant mode. Default: -1.
 * @li active_expert_range: Optional parameter. Type is:ListInt. Like [expert_start, expert_end].
                            expert_start must be greater than or equal to 0, expert_end must be less than or equal to 10240,
                            expert_start must be less than expert_end.
 * @li row_idx_type: Optional parameter. Type is:Int. The value is 0(gather) or 1(scatter). Default: 0.
 */
REG_OP(MoeInitRoutingV3)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(expert_idx, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(expanded_x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OUTPUT(expanded_row_idx, TensorType({DT_INT32}))
    .OUTPUT(expert_tokens_count_or_cumsum, TensorType({DT_INT64}))
    .OUTPUT(expanded_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .ATTR(active_num, Int, -1)
    .ATTR(expert_capacity, Int, -1)
    .ATTR(expert_num, Int, -1)
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(expert_tokens_num_type, Int, 0)
    .ATTR(expert_tokens_num_flag, Bool, false)
    .ATTR(quant_mode, Int, -1)
    .ATTR(active_expert_range, ListInt, {})
    .ATTR(row_idx_type, Int, 0)
    .OP_END_FACTORY_REG(MoeInitRoutingV3)


/**
* @brief In MoE computation, the final step involves processing and merging the output results of the MoE FNN.
* @par Inputs:
* @li expanded_x: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li x1: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li x2: An optional Tensor. Type is:BFloat16, Float16 or Float32.
* @li bias: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li scales: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li expanded_row_idx: A Tensor. Type is:Int32.
* @li expanded_expert_idx: A Tensor. Type is:Int32.
* @par Outputs:
* @li y: A Tensor. Type is:BFloat16, Float16 or Float32.
*/
REG_OP(MoeFinalizeRouting)
    .INPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(expanded_row_idx, TensorType({DT_INT32, DT_INT32, DT_INT32}))
    .INPUT(expanded_expert_idx, TensorType({DT_INT32, DT_INT32, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MoeFinalizeRouting)


/**
* @brief compute softmax and topk for moe input.
* @par Inputs:
* @li x: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li finished: A Tensor. Type is:Bool.
* @par Outputs:
* @li y: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li expert_idx: A Tensor. Type is:Int32.
* @li row_idx: A Tensor. Type is:Int32.
* @par Attributes:
* @li k: Required parameter. Type is:Int32.
*/
REG_OP(MoeGatingTopKSoftmax)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(finished, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(expert_idx, TensorType({DT_INT32}))
    .OUTPUT(row_idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(k, Int)
    .OP_END_FACTORY_REG(MoeGatingTopKSoftmax)


/**
* @brief compute softmax and topk for moe input v2.
* @par Inputs:
* @li x: A Tensor. Type is: BFloat16, Float16 or Float32.
* @li finished: A Tensor. Type is: Bool.
* @par Outputs:
* @li y: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li expert_idx: A Tensor. Type is:Int32.
* @li softmax_result: A Tensor. Type is: Float32.
* @par Attributes:
* @li k: Required parameter. Type is:Int32.
* @li renorm: Optional parameter. Type is:Int32.
* @li output_softmax: Optional parameter. Type is: Bool.
 */
 REG_OP(MoeGatingTopKSoftmaxV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(finished, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(expert_idx, TensorType({DT_INT32}))
    .OUTPUT(softmax_result, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(k, Int)
    .ATTR(renorm, Int)
    .ATTR(output_softmax, Bool)
    .OP_END_FACTORY_REG(MoeGatingTopKSoftmaxV2)
    

/**
* @brief Apply add operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value add by the scale.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachAddScalarInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddScalarInplace)


/**
* @brief Apply add operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are add by the scale
*/
REG_OP(ForeachAddScalar)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddScalar)


/**
* @brief Apply add operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors,
 * meanwhile, this value is also an output, store the value add by the scale.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachAddScalarListInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddScalarListInplace)


/**
* @brief Apply add operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are add by the scales in scalar list
*/
REG_OP(ForeachAddScalarList)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddScalarList)


/**
* @brief Apply add operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value add by the scale.
 * @li x2: Another tensor list containing multiple tensors
 * @li alpha: The elements in x2 should perform multipy with alpha which is a scalar
*/
REG_OP(ForeachAddListInplace)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddListInplace)


/**
* @brief QuantMatmulDequant operator interface implementation.

* @par Inputs
* @li x: A Tensor. Support dtype: float16, dimension must be 2, support format: ND.
* @li quantized_weight: A Tensor. Support dtype: int8, dimension must be 2, support format: ND,NZ.
* @li weight_scale: A Tensor. Support dtype: float32, support format: ND.
* @li bias: A optional Tensor. Support dtype: int32, support format: ND.
* @li x_scale: A optional Tensor. Support dtype: float32, support format: ND.
* @li x_offset: A optional Tensor. Support dtype: float32, support format: ND.
* @li smooth_scale: A optional Tensor. Support dtype: float16, support format: ND.

* @par Attributes
* @li x_quant_mode: dtype: String.
* @li transpose_weight: dtype: Bool.

* @par Outputs
* @li y: A Tensor. Support dtype: float16, support format: ND.
*/
REG_OP(QuantMatmulDequant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(quantized_weight, TensorType({DT_INT8}))
    .INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scale, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(x_quant_mode, String, "pertoken")
    .ATTR(transpose_weight, Bool, true)
    .OP_END_FACTORY_REG(QuantMatmulDequant)


/**
* @brief QuantGroupedMatmulDequant operator interface implementation.

* @par Inputs
* @li x: A Tensor. Support dtype: float16, dimension must be 2, support format: ND.
* @li quantized_weight: A Tensor. Support dtype: int8, dimension must be 3, support format: ND,NZ.
* @li weight_scale: A Tensor. Support dtype: float32, support format: ND.
* @li group_list: A Tensor. Support dtype: int64, support format: ND.
* @li bias: A optional Tensor. Support dtype: int32, support format: ND.
* @li x_scale: A optional Tensor. Support dtype: float32, support format: ND.
* @li x_offset: A optional Tensor. Support dtype: float32, support format: ND.
* @li smooth_scale: A optional Tensor. Support dtype: float16, support format: ND.

* @par Attributes
* @li x_quant_mode: dtype: String.
* @li transpose_weight: dtype: Bool.

* @par Outputs
* @li y: A Tensor. Support dtype: float16, support format: ND.
*/
REG_OP(QuantGroupedMatmulDequant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(quantized_weight, TensorType({DT_INT8}))
    .INPUT(weight_scale, TensorType({DT_FLOAT}))
    .INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scale, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(x_quant_mode, String, "pertoken")
    .ATTR(transpose_weight, Bool, true)
    .OP_END_FACTORY_REG(QuantGroupedMatmulDequant)


/**
* @brief Apply add operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensors
 * @li alpha: The elements in x2 should perform multipy with alpha which is a scalar
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are add by the scales in scalar list
*/
REG_OP(ForeachAddList)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddList)


/**
* @brief Apply sub operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value sub by the scale.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachSubScalarInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSubScalarInplace)


/**
* @brief Apply sub operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are sub by the scale
*/
REG_OP(ForeachSubScalar)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSubScalar)


/**
* @brief Apply sub operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value sub by the scale.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachSubScalarListInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSubScalarListInplace)


/**
* @brief Apply sub operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are sub by the scales in scalar list
*/
REG_OP(ForeachSubScalarList)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSubScalarList)


/**
* @brief Apply sub operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value sub by the scale.
 * @li x2: Another tensor list containing multiple tensors
 * @li alpha: The elements in x2 should perform multipy with alpha which is a scalar
*/
REG_OP(ForeachSubListInplace)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSubListInplace)


/**
* @brief Apply sub operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensors
 * @li alpha: The elements in x2 should perform multipy with alpha which is a scalar
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are sub by the scales in scalar list
*/
REG_OP(ForeachSubList)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSubList)


/**
* @brief Apply mul operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are mul by the scale
*/
REG_OP(ForeachMulScalar)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMulScalar)


/**
* @brief Apply mul operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value mul by the scale.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachMulScalarListInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMulScalarListInplace)


/**
* @brief Apply mul operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are mul by the scales in scalar list
*/
REG_OP(ForeachMulScalarList)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMulScalarList)


/**
* @brief Apply mul operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value mul by the scale.
 * @li x2: Another tensor list containing multiple tensors
*/
REG_OP(ForeachMulListInplace)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMulListInplace)


/**
* @brief Apply mul operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are mul by the scales in scalar list
*/
REG_OP(ForeachMulList)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMulList)


/**
* @brief Apply div operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value div by the scale.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachDivScalarInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachDivScalarInplace)


/**
* @brief Apply div operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are div by the scale
*/
REG_OP(ForeachDivScalar)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachDivScalar)

/**
* @brief Apply div operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value div by the scale.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachDivScalarListInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachDivScalarListInplace)


/**
* @brief Apply div operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are div by the scales in scalar list
*/
REG_OP(ForeachDivScalarList)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachDivScalarList)


/**
* @brief Apply Div operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value Div by the scale.
 * @li x2: Another tensor list containing multiple tensors
*/
REG_OP(ForeachDivListInplace)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachDivListInplace)


/**
* @brief Apply div operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are div by the scales in scalar list
*/
REG_OP(ForeachDivList)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachDivList)


/**
* @brief Apply maximum operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value maximum with the scale.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachMaximumScalarInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMaximumScalarInplace)


/**
* @brief Apply maximum operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are maximum with the scale
*/
REG_OP(ForeachMaximumScalar)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMaximumScalar)


/**
* @brief Apply maximum operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value maximum with the scale.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachMaximumScalarListInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMaximumScalarListInplace)


/**
* @brief Apply maximum operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are maximum with the scales in scalar list
*/
REG_OP(ForeachMaximumScalarList)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMaximumScalarList)


/**
* @brief Apply maximum operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value maximum with the scale.
 * @li x2: Another tensor list containing multiple tensors
*/
REG_OP(ForeachMaximumListInplace)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMaximumListInplace)


/**
* @brief Apply maximum operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are maximum with the scales in scalar list
*/
REG_OP(ForeachMaximumList)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMaximumList)


/**
* @brief Apply minimum operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value minimum with the scale.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachMinimumScalarInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMinimumScalarInplace)


/**
* @brief Apply minimum operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are minimum with the scale
*/
REG_OP(ForeachMinimumScalar)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMinimumScalar)


/**
* @brief Apply minimum operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value minimum with the scale.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachMinimumScalarListInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMinimumScalarListInplace)


/**
* @brief Apply minimum operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are minimum with the scales in scalar list
*/
REG_OP(ForeachMinimumScalarList)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMinimumScalarList)


/**
* @brief Apply minimum operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value minimum with the scale.
 * @li x2: Another tensor list containing multiple tensors
*/
REG_OP(ForeachMinimumListInplace)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMinimumListInplace)


/**
* @brief Apply minimum operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are minimum with the scales in scalar list
*/
REG_OP(ForeachMinimumList)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMinimumList)


/**
* @brief Apply power operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value power with the scale.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachPowScalarInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachPowScalarInplace)


/**
* @brief Apply power operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scale
*/
REG_OP(ForeachPowScalar)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachPowScalar)


/**
* @brief Apply power operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value power with the scale.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachPowScalarListInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachPowScalarListInplace)


/**
* @brief Apply power operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scales in scalar list
*/
REG_OP(ForeachPowScalarList)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachPowScalarList)


/**
* @brief Apply power operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value power with the scale.
 * @li x2: Another tensor list containing multiple tensors
*/
REG_OP(ForeachPowListInplace)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachPowListInplace)


/**
* @brief Apply power operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scales in scalar list
*/
REG_OP(ForeachPowList)
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachPowList)


/**
* @brief Apply abs operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachAbsInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAbsInplace)


/**
* @brief Apply abs operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the abs value of the x
*/
REG_OP(ForeachAbs)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAbs)


/**
* @brief Apply arc cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachACosInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachACosInplace)


/**
* @brief Apply arc cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the arc cos value of the x
*/
REG_OP(ForeachACos)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachACos)


/**
* @brief Apply arc sin operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachASinInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachASinInplace)


/**
* @brief Apply arc sin operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the arc sin value of the x
*/
REG_OP(ForeachASin)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachASin)


/**
* @brief Apply arc tan operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachATanInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachATanInplace)


/**
* @brief Apply arc tan operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the arc tan value of the x
*/
REG_OP(ForeachATan)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachATan)


/**
* @brief Apply cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachCosInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachCosInplace)


/**
* @brief Apply cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the cos value of the x
*/
REG_OP(ForeachCos)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachCos)


/**
* @brief Apply cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachSinInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSinInplace)


/**
* @brief Apply cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the cos value of the x
*/
REG_OP(ForeachSin)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSin)


/**
* @brief Apply tan operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachTanInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachTanInplace)


/**
* @brief Apply tan operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the tan value of the x
*/
REG_OP(ForeachTan)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachTan)


/**
* @brief Apply cosh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachCoshInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachCoshInplace)


/**
* @brief Apply cosh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the cosh value of the x
*/
REG_OP(ForeachCosh)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachCosh)


/**
* @brief Apply sinh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachSinhInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSinhInplace)


/**
* @brief Apply sinh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the sinh value of the x
*/
REG_OP(ForeachSinh)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSinh)


/**
* @brief Apply tanh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachTanhInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachTanhInplace)


/**
* @brief Apply tanh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the tanh value of the x
*/
REG_OP(ForeachTanh)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachTanh)


/**
* @brief Apply sqrt operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachSqrtInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSqrtInplace)


/**
* @brief Apply sqrt operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the sqrt value of the x
*/
REG_OP(ForeachSqrt)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSqrt)


/**
* @brief Apply neg operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachNegInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachNegInplace)


/**
* @brief Apply neg operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the neg value of the x
*/
REG_OP(ForeachNeg)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachNeg)


/**
* @brief Apply exp operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachExpInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachExpInplace)


/**
* @brief Apply exp operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the exp value of the x
*/
REG_OP(ForeachExp)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachExp)


/**
* @brief Apply expm1 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachExpm1Inplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachExpm1Inplace)


/**
* @brief Apply expm1 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the expm1 value of the x
*/
REG_OP(ForeachExpm1)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachExpm1)


/**
* @brief Apply log operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachLogInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLogInplace)


/**
* @brief Apply log operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log value of the x
*/
REG_OP(ForeachLog)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLog)


/**
* @brief Apply log2 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachLog2Inplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLog2Inplace)


/**
* @brief Apply log2 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log2 value of the x
*/
REG_OP(ForeachLog2)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLog2)


/**
* @brief Apply log10 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachLog10Inplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLog10Inplace)


/**
* @brief Apply log10 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log10 value of the x
*/
REG_OP(ForeachLog10)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLog10)


/**
* @brief Apply log1p operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachLog1pInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLog1pInplace)


/**
* @brief Apply log1p operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log1p value of the x
*/
REG_OP(ForeachLog1p)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLog1p)


/**
* @brief Apply reciprocal operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachReciprocalInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachReciprocalInplace)


/**
* @brief Apply reciprocal operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the reciprocal value of the x
*/
REG_OP(ForeachReciprocal)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachReciprocal)


/**
* @brief Apply zero operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachZeroInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachZeroInplace)


/**
* @brief Apply sigmoid operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachSigmoidInplace)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSigmoidInplace)


/**
* @brief Apply sigmoid operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the sigmoid value of the x
*/
REG_OP(ForeachSigmoid)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSigmoid)

/**
* @brief Performs the backpropagation of ROI Align Rotated . \n

* @par Inputs:
* @li x: A tensor of type float32, describing the feature_map.
* @li rois: A tensor of type float32, with shape(n, 6) with each roi decoded as
*     (batch_index, center_x, center_y, w, h, angle). The angle is in radian.

* @par Attributes:
* @li pooled_h: A required int32, specifying the pooled H. Must be greater than 0.
* @li pooled_w: A required int32, specifying the pooled W. Must be greater than 0.
* @li spatial_scale: An required scaling factor for mapping the input coordinates
*     to the ROI coordinates.
* @li sampling_ratio: An required number of inputs samples to take for each output sample.
*     0 to take samples densely for current models.
* @li aligned: A required bool, if False, use the legacy implementation.
*     If True, align the results more perfectly. Default: True.
* @li clockwise: A required bool, if True, the angle in each proposal follows a clockwise
*     fashion in image space, Otherwise, the angle is counterclockwise. Default: False. \n

* @par Outputs:
* @li y: A tensor of type float32, describing the result. \n

* @par Third-party framework compatibility
* It has a corresponding operator in MMCV.
*/
REG_OP(RoiAlignRotatedGrad)
.INPUT(x_grad, TensorType({DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(y_grad_shape, ListInt)
    .REQUIRED_ATTR(pooled_h, Int)
    .REQUIRED_ATTR(pooled_w, Int)
    .REQUIRED_ATTR(spatial_scale, Float)
    .ATTR(sampling_ratio, Int, 0)
    .ATTR(aligned, Bool, true)
    .ATTR(clockwise, Bool, false)
    .OUTPUT(y_grad, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(RoiAlignRotatedGrad)

/**
* @brief Binary finds the position of the last row processed by each expert in the sorted_experts array.
* @par Inputs:
* @li sorted_experts: A Tensor. Type is:Int32.
* @par Outputs:
* @li total_rows_before_expert: A Tensor. Type is:Int32.
* @par Attributes:
* @li num_experts: Required parameter. Type is:Int. The value must be more than 0 and less than 2147483647.
*/
REG_OP(MoeComputeExpertTokens)
    .INPUT(sorted_experts, TensorType({DT_INT32}))
    .OUTPUT(total_rows_before_expert, TensorType({DT_INT32}))
    .REQUIRED_ATTR(num_experts, Int)
    .OP_END_FACTORY_REG(MoeComputeExpertTokens)

/**
* @brief Fusion op for FFN.
* @par Inputs:
* ten inputs, including:
* @li x: A matrix Tensor. The type support int8, float16, bf16.
* @li weight1: A matrix Tensor. The type support int8, float16, bf16, int4.
* @li weight2: A matrix Tensor. The type support int8, float16, bf16, int4.
* @li expert_tokens: A matrix Tensor. The type support int64.
* @li bias1: A matrix Tensor. The type support int32, float16, float32.
* @li bias2: A matrix Tensor. The type support int32, float16, float32.
* @li scale: A matrix Tensor. The type support float32.
* @li offset: A matrix Tensor. The type support float32.
* @li deq_scale1: A matrix Tensor. The type support uint64, bf16.
* @li deq_scale2: A matrix Tensor. The type support uint64, bf16.
* @li antiquant_scale1: A matrix Tensor. The type support float16, bf16.
* @li antiquant_scale2: A matrix Tensor. The type support float16, bf16.
* @li antiquant_offset1: A matrix Tensor. The type support float16, bf16.
* @li antiquant_offset2: A matrix Tensor. The type support float16, bf16.

* @par Attributes:
* @li activation: A string. The type of activation.
* @li inner_precise: A int. 0, fp16 high precision. 1, high performance. Default value: 0
* @li output_dtype: A int. 0, dtype float16. 1, dtype bf16. -1, default dtype. Default value: -1
* @li tokens_index_flag: A boolean. Indicates whether to use index token list. Default value: false
*
* @par Outputs:
* y: A matrix Tensor. The type support float16, bf16. \n
*/
REG_OP(FFN)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16}))
    .INPUT(weight1, TensorType({DT_INT8, DT_FLOAT16}))
    .INPUT(weight2, TensorType({DT_INT8, DT_FLOAT16}))
    .OPTIONAL_INPUT(expert_tokens, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias1, TensorType({DT_INT32, DT_FLOAT16}))
    .OPTIONAL_INPUT(bias2, TensorType({DT_INT32, DT_FLOAT16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(antiquant_scale1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(antiquant_scale2, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(antiquant_offset1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(antiquant_offset2, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(activation, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(output_dtype, Int, -1)
    .ATTR(tokens_index_flag, Bool, false)
    .OP_END_FACTORY_REG(FFN)


/**
* @brief Function GroupedMatmul. \n
* @par Inputs:
* @li x: A Tensor List.
* @li weight: A Tensor List of weight.
* @li bias: A Tensor List of bias.
* @li scale: A Tensor List of scale.
* @li offset: A Tensor List of offset.
* @li antiquant_scale: A Tensor List of antiquant_scale.
* @li antiquant_offset: A Tensor List of antiquant_offset.
* @li group_list: a Tensor.

* @par Attributes:
* @li split_item: A int.
* @li dtype: A int, User is not visible.
* @li transpose_weight: A bool, User is not visible.
*
* @par Outputs:
* y: A Tensor List.
*/
REG_OP(GroupedMatmul)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .DYNAMIC_INPUT(scale, TensorType({DT_UINT64}))
    .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))
    .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(split_item, Int, 0)
    .ATTR(dtype, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(transpose_x, Bool, false)
    .ATTR(group_type, Int, -1)
    .ATTR(tuning_config, ListInt, {0})
    .OP_END_FACTORY_REG(GroupedMatmul)


/**
 * @brief The fusion operator of RMSNorm, RotaryPositionEmbedding and Update KVCache.
 *
 * @par Inputs:
 * @li kv: A Tensor. The type support float16. Format: ND.
 * @li gamma: A Tensor, used in RMS Norm. The type support float16. Format: ND.
 * @li cos: A Tensor, from position embedding. The type support float16. Format: ND.
 * @li sin: A Tensor, from position embedding. The type support float16. Format: ND.
 * @li index: A Tensor. The type support int64. Format: ND.
 * @li k_cache: A Tensor. The type support float16. Format: ND.
 * @li v_cache: A Tensor. The type support float16. Format: ND.
 *
 * @par Outputs:
 * @li k_cache: A Tensor. The type support float16. Format: ND.
 * @li v_cache: A Tensor. The type support float16. Format: ND.
 *
 * @par Attributes:
 * epsilon: A float32. The epsilon value for RMSNorm. Default: 1e-5.
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(KvRmsNormRopeCache)
    .INPUT(kv, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(cos, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(sin, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(index, TensorType({DT_INT64}))
    .INPUT(k_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(ckv_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(k_rope_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(c_kv_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(k_rope_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(c_kv_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(v, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(k_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(ckv_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(k_rope, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(c_kv, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-5)
    .ATTR(cache_mode, String, 'Norm')
    .ATTR(is_output_kv, Bool, false)
    .OP_END_FACTORY_REG(KvRmsNormRopeCache)


/**
 * @brief The fusion operator of Interleave RotaryPositionEmbedding.
 *
 * @par Inputs:
 * @li x: A Tensor. The type support float16. Format: ND.
 * @li cos: A Tensor. The type support float16. Format: ND.
 * @li sin: A Tensor. The type support float16. Format: ND.
 *
 * @par Outputs:
 * y: A Tensor. The type support float16. Format: ND.
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(InterleaveRope)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(InterleaveRope)


/**
 * @brief compute softmax and topk for moe input.
 * @par Inputs:
 * @li x: A 2D Tensor. Type is:BFloat16, Float16 or Float32. Format support ND.
 * @li bias: A 2D Tensor. Type is:BFloat16, Float16 or Float32. Format support ND.
 * @par Outputs:
 * @li y: A Tensor. Type is:BFloat16, Float16 or Float32. The data type must be the same as that of x.
     The size of the non-1 axis must be the same as that of the corresponding axis of x.
        The size of the -1 axis must be the same as that of k. Format support ND.
* @li expert_idx: A Tensor. Type is:Int32. The shape must be the same as that of y. Format support ND.
* @li out: A Tensor. Type is:Float32. The shape must be the same as that of x. Format support ND.
* @par Attributes:
* @li k: Required parameter. Type is:Int32. The value must greater than 0 and less than or equal to the size
        of the -1 axis of x, and k must not greater than 1024.
* @li k_group: Optional parameter. Type is:Int32.
* @li group_count: Optional parameter. Type is:Int32.
* @li group_select_mode: Optional parameter. Type is:Int32.
* @li renorm: Optional parameter. Type is:Int32.
* @li out_flag: Optional parameter. Type is:Bool.
* @li routed_scaling_factor: Optional parameter. Type is:Float32.
* @li eps: Optional parameter. Type is:Float32.
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(MoeGatingTopK)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(expert_idx, TensorType({DT_INT32}))
    .OUTPUT(out, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(k, Int)
    .ATTR(k_group, Int, 1)
    .ATTR(group_count, Int, 1)
    .ATTR(group_select_mode, Int, 0)
    .ATTR(renorm, Int, 0)
    .ATTR(norm_type, Int, 0)
    .ATTR(out_flag, Bool, false)
    .ATTR(routed_scaling_factor, Float, 1.0)
    .ATTR(eps, Float, 1e-20)
    .OP_END_FACTORY_REG(MoeGatingTopK)


/**
* @brief Combine Dequant + Swiglu + Quant.

* @par Inputs:
* Seven inputs, including:
* @li x: A Tensor. Type is:DT_INT32. Shape is (X..., H), Dim must > 2, and H must be even.
* @li weight_scale: A optional tensor. Type is:DT_FLOAT. Shape is (1..., H).
* @li activation_scale: A optional tensor. Type is:DT_FLOAT. Shape is  (X..., 1).
* @li bias: A optional tensor. Type is:DT_FLOAT. Shape is (X..., H).
* @li quant_scale: A optional tensor. Type is:DT_FLOAT. Shape is (1..., H).
* @li quant_offset: A optional tensor. Type is:DT_FLOAT. Shape is (1..., H).
* @li group_index: A optional tensor. Type is:DT_INT64, DT_INT32. Shape is (1,). mean group input. \n

* @par Outputs:
* @li y: A Tensor. Type is:DT_INT8.
* @li scale: A Tensor. Type is:DT_FLOAT.

* @par Attributes:
* @li activate_left: Type is: Bool.
* The swi activate_left algorithm to use:
*     'false'(activate right) or 'true'(activate left), defalut is 'false'(activate right).
* @li quant_mode: Type is: String. The quant mode to use: 'static' or 'dynamic', defalut is 'static'.
* @li swiglu_mode: Type is int. Optional parameter, default is 0. The SWIGLU computation mode to use: 
*     '0' (default) for standard SWIGLU, '1' for a variant using odd-even blocking, which requires support for clamp_limit, activation coefficient, and bias.
* @li clamp_limit: Type is float. Optional parameter, default is 7.0. The threshold limit for SWIGLU input.
* @li glu_alpha: Type is float. Optional parameter, default is 1.702. The activation coefficient for the GLU activation function.
* @li glu_bias: Type is float. Optional parameter, default is 1.0. The bias applied during SWIGLU linear computation.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DequantSwigluQuant)
    .INPUT(x, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(activate_left, Bool, false)
    .ATTR(quant_mode, String, "static")
    .ATTR(swiglu_mode, int, 0)
    .ATTR(clamp_limit, Float, 7.0)
    .ATTR(glu_alpha, Float, 1.702)
    .ATTR(glu_bias, Float, 1.0)
    .OP_END_FACTORY_REG(DequantSwigluQuant)


/**
* @brief Activation function of SwiGlu with clipping.

* @par Inputs:
* Two inputs, including:
* @li x: A tensor. Type is bfloat16, float16, float32.
* @li group_index: An optional tensor. Shape is (N,). Type is int64. 

* @par Outputs:
* one output, including:
* y: A tensor. Type is bfloat16, float16, float32.

* @par Attributes:
* Five attributes, including:
* @li dim: An optional int. The dimension to be split, value in [-xDim, xDim-1], default is -1.
* @li alpha: An optional float. The activation coefficient for the GLU activation function, default is 1.702.
* @li limit: An optional float. The threshold limit for SWIGLU input, default is 7.0.
* @li bias: An optional float. The bias applied during SWIGLU linear computation, default is 1.0.
* @li interleaved: An optional bool. The way of splitting x: true for interleaved splitting, false for front-back splitting, default is true.

* @attention Constraints:
* The dim dimension of x must be divisible by 2, and the dim dimension of y must be equal to the dim dimension of x divided by 2.
*/
REG_OP(ClippedSwiglu)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(dim, Int, -1)
    .ATTR(alpha, Float, 1.702)
    .ATTR(limit, Float, 7.0)
    .ATTR(bias, Float, 1.0)
    .ATTR(interleaved, Bool, true)
    .OP_END_FACTORY_REG(ClippedSwiglu)


/**
* @brief  Function AttentionUpdate.

* @par Inputs:
* Two inputs, including:
* @li lse: Tensor list. Type is float32. The input of lse.
* @li go: Tensor list. Type is float32, float16, bfloat16. The input of attentionout.

* @par Outputs:
* Two outputs, including:
* @li output: A tensor. Type is float32, float16, bfloat16.
* @li lse_m: A tensor. Type is float32.  

* @par Attributes:
* Two attributes, including:
* @li update_type: An int. The update type, value is 0 or 1. 0 means the output of lse_m is invalid, 1 means valid.
* @li sp: An int. The sp num, value is [1..16].
*/
REG_OP(AttentionUpdate)
    .DYNAMIC_INPUT(lse, TensorType({DT_FLOAT}))
    .DYNAMIC_INPUT(go, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(lse_m, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(update_type, Int)
    .REQUIRED_ATTR(sp, Int)
    .OP_END_FACTORY_REG(AttentionUpdate)


/**
* @brief Function FusedInferAttentionScore.

* @par Inputs:
* @li query: A matrix Tensor. The type support int8, float16, bf16.
* @li key: A matrix Tensor. The type support int8, float16, bf16.
* @li value: A matrix Tensor. The type support int8, float16, bf16.
* @li pse_shift: A matrix Tensor. The type support float16, bf16.
* @li atten_mask: A matrix Tensor. The type support float16, bool, uint8, int8.
* @li actual_seq_lengths: A Tensor. The type support INT64.
* @li actual_seq_lengths_kv: A Tensor. The type support INT64.
* @li dequant_scale1: A Tensor. The type support UINT64.
* @li quant_scale1: A Tensor. The type support float32.
* @li dequant_scale2: A Tensor. The type support UINT64.
* @li quant_scale2: A Tensor. The type support float32.
* @li quant_offset2: A Tensor. The type support float32.
* @li antiquant_scale: A Tensor. The type support float16, bf16.
* @li antiquant_offset: A Tensor. The type support float16, bf16.
* @li block_table: An int.

* @par Attributes:
* @li num_heads: An int. The number of the heads.
* @li scale: A float. The scale value. Default: 1.0.
* @li pre_tokens: An int. Previous tokens. Default: 2147483647.
* @li next_tokens: An int. Next tokens. Default: 2147483647.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD", "BSND", "NSD", "SH"]. Default: "BSH".
* @li num_key_value_heads: key value num heads. Default: 0.
* @li sparse_mode: sparse mode. Default: 0.
* @li inner_precise: An int. 0, float16 high precision. 1, high performance. Default: 0.
* @li block_size: An int. Default: 0.

* @par Outputs:
* attention_out: A matrix Tensor. The type support float16, float32, int8, bf16. \n
*/
REG_OP(FusedInferAttentionScore)
    .INPUT(query, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(key, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(value, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_UINT8, DT_INT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(key_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_shared_prefix, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(value_shared_prefix, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(actual_shared_prefix_len, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope_antiquant_scale, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale_query, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(learnable_sink, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_BF16}))
    .OUTPUT(softmax_lse, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale, Float, 1.0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 0)
    .ATTR(block_size, Int, 0)
    .ATTR(antiquant_mode, Int, 0)
    .ATTR(softmax_lse_flag, Bool, false)
    .ATTR(key_antiquant_mode, Int, 0)
    .ATTR(value_antiquant_mode, Int, 0)
    .ATTR(query_quant_mode, Int, 0)
    .ATTR(pse_type, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .OP_END_FACTORY_REG(FusedInferAttentionScore)


/**
* @brief Function FusedInferAttentionScore.

* @par Inputs:
* @li key: A matrix Tensor. The type support int8, uint8, hifloat8, float8_e5m2, float8_e4m3fn, int16, uint16, float16, bf16, int32, uint32, float.
* @li value: A matrix Tensor. The type support int8, uint8, hifloat8, float8_e5m2, float8_e4m3fn, int16, uint16, float16, bf16, int32, uint32, float.
* @li slot_mapping: A matrix Tensor. The type support int32, int64.
* @li compress_lens: A matrix Tensor. The type support int32, int64.
* @li compress_seq_offset: A Tensor. The type support int32, int64.
* @li seq_lens: A Tensor. The type support int32, int64.

* @par Attributes
* @li cache_mode: A string. The data format of key_cache and value_cache, "Norm" means ND.
* @li scatter_mode: An optional attribute. Describing the format of cache. Defaults to "None". 
* @li strides: An optional attribute. A list of 2 integers. The stride of the key and value, its' shape is [stride_k, stride_v].
* @li offsets: An optional attribute. A list of 2 integers. The offsets of the key and value, its' shape is [offset_k, offset_v].

* @par Outputs:
* @li key_cache: A matrix Tensor. The type support int8, uint8, hifloat8, float8_e5m2, float8_e4m3fn, int16, uint16, float16, bf16, int32, uint32, float.
* @li value_cache: A matrix Tensor. The type support int8, uint8, hifloat8, float8_e5m2, float8_e4m3fn, int16, uint16, float16, bf16, int32, uint32, float.
*/
REG_OP(ScatterPaKvCache)
    .INPUT(key, "T")
    .INPUT(key_cache, "T")
    .INPUT(slot_mapping, TensorType::IndexNumberType())
    .INPUT(value, "T")
    .INPUT(value_cache, "T")
    .OPTIONAL_INPUT(compress_lens, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(compress_seq_offset, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(seq_lens, TensorType::IndexNumberType())
    .OUTPUT(key_cache, "T")
    .OUTPUT(value_cache, "T")
    .ATTR(cache_mode, String, "Norm")
    .ATTR(scatter_mode, String, "None")
    .ATTR(strides, ListInt, {1,1})
    .ATTR(offsets, ListInt, {0,0})
    .DATATYPE(T, TensorType({DT_INT8, DT_UINT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                             DT_INT16, DT_UINT16, DT_FLOAT16, DT_BF16, DT_INT32, DT_UINT32, DT_FLOAT}))
    .OP_END_FACTORY_REG(ScatterPaKvCache)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
