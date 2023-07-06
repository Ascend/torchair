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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PATTERN_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PATTERN_H_
#include <map>
#include <string>
#include <vector>
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"

namespace fe {
extern const int64_t TBE_FUSION_OP_NUM_MAX;
extern const int64_t TBE_PATTERN_NUM_MAX;
extern const int64_t TBE_PATTERN_NUM_NONE;
extern const int64_t TBE_PATTERN_NUM_DEFAULT;
extern const int64_t TBE_OUTPUT_BRANCH_DEFAULT;
extern const int64_t TBE_OUTPUT_BRANCH_SINGLE;
extern const int64_t TBE_OUTPUT_BRANCH_MULTI;
extern const int64_t TBE_PATTERN_GROUPID_INVALID;
extern const int32_t TBE_OUTPUT_MAX_NUM_LIMIT;

enum SkipStatus { DISABLED = 0, AVAILABLE = 1, SKIPPED = 2 };

enum ShapeTypeRule { IGNORE_SHAPE_TYPE = 0, ONLY_SUPPORT_STATIC, ONLY_SUPPORT_DYNAMIC };

enum class PatternRelation { RELATIVE_POSITION_CONSISTENT = 0 };

extern const std::map<ShapeTypeRule, const std::string> kShapeTypeRuleToStr;

struct BufferFusionOpDesc {
  std::string desc_name;                       // description name
  std::vector<std::string> types;             // description type
  std::vector<BufferFusionOpDesc *> inputs;   // all input op
  std::vector<BufferFusionOpDesc *> outputs;  // all output op
  int64_t out_branch_type;                      // out desc type, 1:single, 2: multi
  int64_t repeate_min;                         // opdesc min repeat num
  int64_t repeate_max;                         // opdesc max repeat num
  int64_t repeate_curr;                        // opdesc current repeat num
  bool match_status;
  bool not_pattern;
  int64_t group_id;  // record desc groupid, need one desc matched at least in
                    // the same group
  std::vector<ShapeTypeRule> shape_type_rules;
  bool ignore_input_num;
  bool ignore_output_num;
  bool is_allow_series;  // whether the nodes with the same pattern can be series in match graph
  int32_t output_max_limit;
  // used for two connected op, first opdesc has optional multiple nodes and
  // ignore_output_num is true, second opdesc is same pattern type and
  // out_branch_type is TBE_OUTPUT_BRANCH_MULTI
  std::map<int64_t, SkipStatus> multi_output_skip_status;
  std::vector<std::pair<const BufferFusionOpDesc *, PatternRelation>> relations;
};

struct MappingCmpKey {
  bool operator() (const BufferFusionOpDesc *key1, const BufferFusionOpDesc *key2) const {
    return (key1->desc_name) < (key2->desc_name);
  }
};
using BufferFusionMapping = std::map<const BufferFusionOpDesc *, std::vector<ge::NodePtr>, MappingCmpKey>;
using BufferFusionNodeDescMap = std::unordered_map<ge::NodePtr, BufferFusionOpDesc *>;

class BufferFusionPattern {
 public:
  explicit BufferFusionPattern(std::string name = "", int64_t op_max_count = TBE_FUSION_OP_NUM_MAX);

  virtual ~BufferFusionPattern();

  /*
   * types vector use one ShapeTypeRule
   */
  BufferFusionPattern &AddOpDesc(const std::string &desc_name, const std::vector<std::string> &types,
                                 const int64_t repeat_min = TBE_PATTERN_NUM_DEFAULT,
                                 const int64_t repeat_max = TBE_PATTERN_NUM_DEFAULT,
                                 const int64_t group_id = TBE_PATTERN_GROUPID_INVALID,
                                 const ShapeTypeRule shape_type_rule = ONLY_SUPPORT_STATIC,
                                 const bool not_pattern = false, const bool is_allow_series = true);

/**
 * add node desc
 * @param desc_name
 * @param types
 * @param repeat_min
 * @param repeat_max
 * @param is_allow_series
 * @return ref
 */
  BufferFusionPattern &AddOpDesc(const std::string &desc_name, const std::vector<std::string> &types,
                                 const int64_t repeat_min, const int64_t repeat_max, const bool is_allow_series);

  /*
   * types vector use ShapeTypeRule vector, and size should be same or ShapeTypeRule size equal 1
   */
  BufferFusionPattern &AddOpDescTypeRules(const std::string &desc_name, const std::vector<std::string> &types,
                                          const int64_t repeat_min, const int64_t repeat_max, const int64_t group_id,
                                          const std::vector<ShapeTypeRule> &shape_type_rules,
                                          const bool not_pattern = false, const bool is_allow_series = true);

  BufferFusionPattern &SetOutputs(const std::string &desc_name, const std::vector<std::string> &output_ids,
                                  int64_t relation = TBE_OUTPUT_BRANCH_SINGLE, bool ignore_input_num = false,
                                  bool ignore_output_num = false, int32_t output_max_limit = TBE_OUTPUT_MAX_NUM_LIMIT);

  BufferFusionPattern &SetHead(const std::vector<std::string> &head_ids);

  /**
   * add node desc
   * @param desc_name
   * @param types
   * @param repeat_min
   * @param repeat_max
   * @param is_allow_series
   * @return ref
   */
  BufferFusionPattern &SetRelation(const std::string &src_desc_name, const std::string &dst_desc_name,
                                   const PatternRelation pattern_relation);

  const std::string& GetName() const;
  int64_t GetOpMaxCount() const;
  const std::vector<BufferFusionOpDesc *>& GetOpDescs() const;
  const std::vector<BufferFusionOpDesc *>& GetHead() const;
  int64_t GetErrorCnt() const;
  void SetGraphModType(int64_t graph_mod_type);
  int64_t GetGraphModType() const;
  bool GetOutputs(BufferFusionOpDesc *op_desc, std::vector<BufferFusionOpDesc *> &outputs, bool ignore_repeat = false);

 private:
  bool IsOpDescValid(const std::string &desc_name, int64_t repeat_min, int64_t repeat_max) const;
  bool IsShapeRulesSizeValid(const size_t &types_size, const size_t &rules_size) const;
  BufferFusionOpDesc *GetOpDesc(const std::string &desc_name) const;
  void UpdateSkipStatus(const BufferFusionOpDesc *op_desc) const;

  std::string name_;
  int64_t op_max_count_;
  std::vector<BufferFusionOpDesc *> ops_;
  std::map<std::string, BufferFusionOpDesc *> op_map_;
  std::vector<BufferFusionOpDesc *> head_;
  int64_t error_count_;
  // 0: this pattern will not modify graph(default)
  // 1: this pattern will modify graph
  int64_t graph_mod_type_;
};
}  // namespace fe
#endif  // INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PATTERN_H_
