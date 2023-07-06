/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_COMMON_LXFUSION_JSON_UTIL_H_
#define FUSION_ENGINE_INC_COMMON_LXFUSION_JSON_UTIL_H_

#include "common/aicore_util_types.h"
#include "common/aicore_util_attr_define.h"
#include "graph_optimizer/fusion_common/op_slice_info.h"
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "graph/compute_graph.h"

namespace fe {
using ToOpStructPtr = std::shared_ptr<fe::ToOpStruct_t>;
using L2FusionInfoPtr = std::shared_ptr<fe::TaskL2FusionInfo_t>;
extern const std::string L1_FUSION_TO_OP_STRUCT;
extern const std::string L2_FUSION_TO_OP_STRUCT;

Status GetL1InfoFromJson(ge::OpDescPtr op_desc_ptr);

Status GetL2InfoFromJson(ge::OpDescPtr op_desc_ptr);

Status GetTaskL2FusionInfoFromJson(ge::OpDescPtr op_desc_ptr);

Status ReadGraphInfoFromJson(ge::ComputeGraph &graph);

Status WriteGraphInfoToJson(ge::ComputeGraph &graph);

Status ReadOpSliceInfoFromJson(ge::ComputeGraph &graph);

Status WriteOpSliceInfoToJson(ge::ComputeGraph &graph);

void SetOpSliceInfoToJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_str);

void SetFusionOpSliceInfoToJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_str);

void GetOpSliceInfoFromJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_str);

void GetFusionOpSliceInfoFromJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_str);

void GetL2ToOpStructFromJson(ge::OpDescPtr &op_desc_ptr, ToOpStructPtr &l2_info_ptr);

void GetL1ToOpStructFromJson(ge::OpDescPtr &op_desc_ptr, ToOpStructPtr &l1_info_ptr);

void GetStridedToOpStructFromJson(ge::OpDescPtr& op_desc_ptr, ToOpStructPtr& strided_info_ptr,
                                  const string& pointer_key, const string& json_key);

L2FusionInfoPtr GetL2FusionInfoFromJson(ge::OpDescPtr &op_desc_ptr);

void SetL2FusionInfoToNode(ge::OpDescPtr &op_desc_ptr, L2FusionInfoPtr &l2_fusion_info_ptr);

void SetStridedInfoToNode(ge::OpDescPtr& op_desc_ptr, ToOpStructPtr& strided_info_ptr,
                          const string& pointer_key, const string& json_key);
} // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_LXFUSION_JSON_UTIL_H_
