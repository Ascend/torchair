/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file all_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ALL_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ALL_OPS_H_

#include "aipp.h"
#include "arithmetic.h"
#include "array.h"
#include "array_ops.h"
#include "audio.h"
#include "audio_ops.h"
#include "avg_pool_1d_ops.h"
#include "batch_ops.h"
#include "bitwise.h"
#include "bitwise_ops.h"
#include "boosted_trees_ops.h"
#include "candidate_sampling_ops.h"
#include "cast.h"
#include "cluster.h"
#include "combination.h"
#include "compare.h"
#include "complex.h"
#include "condtake_ops.h"
#include "control_flow.h"
#include "control_flow_ops.h"
#include "correlation.h"
#include "ctc_ops.h"
#include "data_flow_ops.h"
#include "deep_md.h"
#include "elewise_calculation_ops.h"
#include "encoding_ops.h"
#include "experiment_ops.h"
#include "fft.h"
#include "functional_ops.h"
#include "get_data_ops.h"
#include "hcom_ops.h"
#include "hpc.h"
#include "hvd_ops.h"
#include "image.h"
#include "image_ops.h"
#include "index.h"
#include "internal_ops.h"
#include "interpolation.h"
#include "ldpc.h"
#include "linalg.h"
#include "linalg_ops.h"
#include "list_ops.h"
#include "logging_ops.h"
#include "logic.h"
#include "lookup.h"
#include "lookup_ops.h"
#include "map_ops.h"
#include "math_ops.h"
#include "matrix_calculation_ops.h"
#include "molecular_dynamics.h"
#include "nn_activation.h"
#include "nn_batch_norm_ops.h"
#include "nn_calculation_ops.h"
#include "nn_detect.h"
#include "nn_detect_ops.h"
#include "nn.h"
#include "nn_list.h"
#include "nn_map.h"
#include "nn_math.h"
#include "nn_matrix_calculation.h"
#include "nn_norm.h"
#include "nn_norm_ops.h"
#include "nn_ops.h"
#include "nn_optimizer.h"
#include "nn_other.h"
#include "nn_pooling.h"
#include "nn_pooling_ops.h"
#include "nn_quantize.h"
#include "nn_recurrent.h"
#include "nn_set.h"
#include "nn_sort.h"
#include "nn_string.h"
#include "nn_training_ops.h"
#include "nonlinear_fuc_ops.h"
#include "no_op.h"
#include "npu_loss_scale_ops.h"
#include "ocr.h"
#include "ocr_ops.h"
#include "ordered_map.h"
#include "outfeed_ops.h"
#include "pad.h"
#include "pad_ops.h"
#include "parsing_ops.h"
#include "quantize_ops.h"
#include "ragged_array_ops.h"
#include "ragged_conversion_ops.h"
#include "ragged_math_ops.h"
#include "randomdsa_ops.h"
#include "random.h"
#include "random_ops.h"
#include "reduce.h"
#include "reduce_ops.h"
#include "resource_variable_ops.h"
#include "rnn.h"
#include "round.h"
#include "rpn_ops.h"
#include "save_ops.h"
#include "sdca_ops.h"
#include "search.h"
#include "segment.h"
#include "selection.h"
#include "selection_ops.h"
#include "set_ops.h"
#include "signal.h"
#include "sparse.h"
#include "sparse_ops.h"
#include "spectral_ops.h"
#include "split_combination_ops.h"
#include "stack.h"
#include "stateful_random_ops.h"
#include "stateless_random_ops.h"
#include "state_ops.h"
#include "string_ops.h"
#include "swap_co_ops.h"
#include "system.h"
#include "target_crop_and_resize.h"
#include "tensor_array.h"
#include "tensor.h"
#include "transcendental.h"
#include "transformation.h"
#include "transformation_ops.h"
#include "vector_search.h"
#include "warp_perspective_ops.h"

#endif  // OPS_BUILT_IN_OP_PROTO_INC_ALL_OPS_H_