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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_FUSION_PATTERN_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_FUSION_PATTERN_H_
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fe {
extern const uint32_t kFuzzyOutIndex;
/** Fusion pattern
 * @ingroup FUSION_PASS_GROUP
 * Describe Pattern of Ops waiting for fusion(Op type, etc)
 */
class FusionPattern {
 public:
  struct OpDesc;
  using OpDescPtr = std::shared_ptr<OpDesc>;
  using OutputMapVecStr = std::initializer_list<std::pair<uint32_t, std::initializer_list<const char*>>>;
  using OutputMapStr = std::initializer_list<std::pair<uint32_t, const char*>>;
  using OutputMapDesc = std::map<uint32_t, std::vector<FusionPattern::OpDescPtr>>;
  /**
   * @ingroup fe
   * @brief description of Ops
   */
  struct OpDesc {
    std::string id;                       // Identifier
    std::vector<std::string> types;  // the Op types of Ops
    std::vector<OpDescPtr> inputs;   // all input Ops
    OutputMapDesc outputs; // all output Ops
    bool repeatable;                 // flag to show if match multiple Ops or not
    bool is_output;                  // flag to show if the op is output node
    bool is_output_fullmatch;        // flag to match all output
    size_t output_size;              // all output size
  };

 public:
  explicit FusionPattern(const std::string name = "");
  ~FusionPattern();

  /** set pattern name
   *
   * @param name pattern name
   * @return FusionPattern
   */
  FusionPattern &SetName(const std::string &name);

  /** add Op description with unknown number of args
   *
   * @param id pattern id
   * @param types op type list
   * @return FusionPattern
   */
  FusionPattern &AddOpDesc(const std::string &id, const std::initializer_list<std::string> &types = {});

  /** add Op description with vector
   *
   * @param id pattern id
   * @param types op type list
   *
   * @return FusionPattern
   */
  FusionPattern &AddOpDesc(const std::string &id, const std::vector<std::string> &types);

  /** set input Ops with unknown number of args
   *
   * @param id pattern id
   *
   * @param input_ids inputs to id op
   *
   * @return FusionPattern
   */
  FusionPattern &SetInputs(const std::string &id, const std::initializer_list<std::string> &input_ids);

  /** set input Ops with unknown number of args
   *
   * @param id pattern id
   *
   * @param input_ids inputs to id op
   *
   * @return FusionPattern
   */
  FusionPattern &SetInputs(const std::string &id, const std::vector<std::string> &input_ids);

  /** set output Ops with unknown number of args
   *
   * @param id pattern id
   *
   * @param output_map output map
   *
   * @param is_fullmatched flag of output full matched
   *
   * @return FusionPattern
   */
  FusionPattern &SetOutputs(const std::string &id, const OutputMapStr &output_map, bool is_fullmatched = true);

  /** set output Ops with unknown number of args
   *
   * @param id pattern id
   *
   * @param output_map output map
   *
   * @param is_fullmatched flag of output full matched
   *
   * @return FusionPattern
   */
  FusionPattern &SetOutputs(const std::string &id, const OutputMapVecStr &output_map, bool is_fullmatched = true);

  /** set output Op
   *
   * @param id pattern id
   *
   * @return FusionPattern
   */
  FusionPattern &SetOutput(const std::string &id);

  /** build pattern and check if error exists
   *
   * @return True or False
   */
  bool Build();

  /** get pattern name
   *
   * @param id pattern id
   *
   * @return fusion pattern name
   */
  const std::string &GetName() const;

  /** get the OpDesc of input Ops (const)
   *
   * @param op_desc op_desc for getting inputs
   *
   * @return op_desc's iniput opdesc list
   */
  static const std::vector<std::shared_ptr<OpDesc>> *GetInputs(const std::shared_ptr<FusionPattern::OpDesc> op_desc);

  /** get the OpDesc of output Ops (const)
   *
   * @param op_desc op_desc for getting outputs
   *
   * @return op_desc's output opdesc map
   */
  static const OutputMapDesc &GetOutputs(const OpDescPtr op_desc);

  /** get the OpDesc of output size
   *
   * @param op_desc op_desc for getting output size
   *
   * @return op_desc's output size
   */
  static size_t GetOutputSize(const OpDescPtr op_desc);

  /** get the OpDesc of output Op
   *
   * @return pattern's output opdesc list
   */
  const std::shared_ptr<FusionPattern::OpDesc> GetOutput() const;

  /** print pattern
   *
   */
  void Dump() const;

  /** get OpDesc based on ID, return nullptr if failed
   *
   * @param id pattern id
   *
   * @return pattern's output opdesc list
   */
  std::shared_ptr<FusionPattern::OpDesc> GetOpDesc(const std::string &id) const;

 private:
  FusionPattern(const FusionPattern &) = default;
  FusionPattern &operator=(const FusionPattern &) = default;

  void SetError();

 private:
  std::string name_;

  std::vector<std::shared_ptr<OpDesc>> ops_;

  std::map<std::string, std::shared_ptr<OpDesc>> op_map_;

  std::shared_ptr<OpDesc> output_;

  bool has_error_;
};
struct CmpKey {
  bool operator() (const std::shared_ptr<FusionPattern::OpDesc> &key1,
      const std::shared_ptr<FusionPattern::OpDesc> &key2) const {
      return (key1->id) < (key2->id);
  }
};
}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_FUSION_PATTERN_H_
