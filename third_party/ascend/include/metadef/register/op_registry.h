/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#ifndef INC_REGISTER_OP_REGISTRY_H_
#define INC_REGISTER_OP_REGISTRY_H_

#include <climits>
#include <set>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>

#include "register/register.h"

namespace domi {
enum class RemoveInputType : uint16_t {
  OMG_MOVE_TYPE_DTYPE = 0,
  OMG_MOVE_TYPE_VALUE = 1,
  OMG_MOVE_TYPE_SHAPE = 2,
  OMG_MOVE_TYPE_FORMAT = 3,
  OMG_MOVE_TYPE_AXIS = 4,
  OMG_MOVE_TYPE_SCALAR_VALUE = 5,
  OMG_REMOVE_TYPE_WITH_COND = 1000,
  OMG_REMOVE_INPUT_WITH_ORIGINAL_TYPE = 1001,
  OMG_INPUT_REORDER = 1002,
};

struct RemoveInputConfigure {
  int32_t inputIdx = INT_MAX;
  std::string attrName;
  RemoveInputType moveType;
  bool attrValue = false;
  std::string originalType;
  std::vector<int32_t> input_order;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpRegistry {
 public:
  static OpRegistry *Instance();

  std::vector<OpRegistrationData> registrationDatas;

  bool Register(const OpRegistrationData &reg_data);

  domi::ImplyType GetImplyType(const std::string &op_type);

  void GetOpTypeByImplyType(std::vector<std::string> &vec_op_type, const domi::ImplyType imply_type);

  domi::ParseParamFunc GetParseParamFunc(const std::string &op_type, const std::string &ori_type);

  domi::ParseParamByOpFunc GetParseParamByOperatorFunc(const std::string &ori_type);

  domi::FusionParseParamFunc GetFusionParseParamFunc(const std::string &op_type, const std::string &ori_type);

  domi::FusionParseParamByOpFunc GetFusionParseParamByOpFunc(const std::string &op_type,
                                                             const std::string &ori_type);

  domi::ParseSubgraphFunc GetParseSubgraphPostFunc(const std::string &op_type);

  Status GetParseSubgraphPostFunc(const std::string &op_type, domi::ParseSubgraphFuncV2 &parse_subgraph_func);

  domi::ImplyType GetImplyTypeByOriOpType(const std::string &ori_optype);

  const std::vector<RemoveInputConfigure> &GetRemoveInputConfigure(const std::string &ori_optype) const;

  bool GetOmTypeByOriOpType(const std::string &ori_optype, std::string &om_type);

  ParseOpToGraphFunc GetParseOpToGraphFunc(const std::string &op_type, const std::string &ori_type);

 private:
  std::unordered_map<std::string, domi::ImplyType> op_run_mode_map_;
  std::unordered_map<std::string, ParseParamFunc> op_parse_params_fn_map_;
  std::unordered_map<std::string, ParseParamByOpFunc> parse_params_by_op_func_map_;
  std::unordered_map<std::string, FusionParseParamFunc> fusion_op_parse_params_fn_map_;
  std::unordered_map<std::string, FusionParseParamByOpFunc> fusion_parse_params_by_op_fn_map_;
  std::unordered_map<std::string, ParseSubgraphFunc> op_types_to_parse_subgraph_post_func_;
  std::unordered_map<std::string, std::vector<RemoveInputConfigure>> remove_input_configure_map_;
  std::map<std::string, std::string> origin_type_to_om_type_;
  std::unordered_map<std::string, ParseOpToGraphFunc> parse_op_to_graph_fn_map_;
  std::unordered_map<std::string, ParseSubgraphFuncV2> op_types_to_parse_subgraph_post_func_v2_;
};
}  // namespace domi
#endif  // INC_REGISTER_OP_REGISTRY_H_
