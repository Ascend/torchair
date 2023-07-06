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

#ifndef INC_GRAPH_AXIS_TYPE_INFO_H_
#define INC_GRAPH_AXIS_TYPE_INFO_H_

#include <vector>
#include <map>
#include <string>

#include "external/graph/ge_error_codes.h"

namespace ge {
using CutInfo = std::pair<int64_t, std::vector<int64_t>>;

enum class AxisType {
  UNSPLIT = 0,
  ELEMENTWISE,
  REDUCESUM,
  REDUCEMAX,
  REDUCEMIN,
  REDUCEMEAN,
  TRANSPOSE,
  SLIDINGWINDOW,
  REDUCEGATHER,
  SLIDINGWINDOWGRAD,
  ELEMENTWITHSHAPEVALUE,
  REDUCEPROD,
};

class AxisTypeInfo {
public:
  AxisTypeInfo() = default;
  ~AxisTypeInfo() = default;
  void SetAxisType(const AxisType axis_type) { axis_type_ = axis_type; }
  AxisType GetAxisType() const { return axis_type_; }
  void AddInputCutInfo(CutInfo &input_cut_info);
  void AddOutputCutInfo(CutInfo &output_cut_info);
  graphStatus GetInputCutInfo(const size_t index, CutInfo &input_cut_info) const;
  graphStatus GetOutputCutInfo(const size_t index, CutInfo &output_cut_info) const;
  const std::vector<CutInfo> &GetRelateInputs() const { return relate_inputs_; }
  const std::vector<CutInfo> &GetRelateOutputs() const { return relate_outputs_; }
  void SetRelateInputs(const std::vector<CutInfo> &inputs_info) { relate_inputs_ = inputs_info; }
  void SetRelateOutputs(const std::vector<CutInfo> &outputs_info) { relate_outputs_ = outputs_info; }
  const std::vector<CutInfo> &GetOriRelateInputs() const { return ori_relate_inputs_; }
  const std::vector<CutInfo> &GetOriRelateOutputs() const { return ori_relate_outputs_; }
  void SetOriRelateInputs(const std::vector<CutInfo> &inputs_info) { ori_relate_inputs_ = inputs_info; }
  void SetOriRelateOutputs(const std::vector<CutInfo> &outputs_info) { ori_relate_outputs_ = outputs_info; }
  void SetAxisTypes(const std::vector<AxisType> &axis_types) { axis_types_ = axis_types; }
  const std::vector<AxisType> &GetAxisTypes() const { return axis_types_; }
  void AddInputValueCutInfo(const CutInfo &cut_info);
  void AddOutputValueCutInfo(const CutInfo &cut_info);
  graphStatus GetInputValueCutInfo(const size_t index, CutInfo &cut_info) const;
  graphStatus GetOutputValueCutInfo(const size_t index, CutInfo &cut_info) const;
  const std::vector<CutInfo> &GetRelateInputValues() const { return relate_input_values_; }
  const std::vector<CutInfo> &GetRelateOutputValues() const { return relate_output_values_; }
  void SetAdditionInfo(const std::map<std::string, std::string> &addition_info) { addition_info_ = addition_info; }
  const std::map<std::string, std::string> &GetAdditionInfo() const { return addition_info_; }

private:
  static graphStatus DoGetCutInfo(const std::vector<CutInfo> &cut_infos, const size_t index, CutInfo &cut_info);

  AxisType axis_type_ = AxisType::UNSPLIT;
  std::vector<CutInfo> relate_inputs_;
  std::vector<CutInfo> relate_outputs_;
  std::vector<AxisType> axis_types_;
  std::vector<CutInfo> relate_input_values_;
  std::vector<CutInfo> relate_output_values_;
  std::vector<CutInfo> ori_relate_inputs_; // backup relate_inputs_
  std::vector<CutInfo> ori_relate_outputs_; // backup relate_outputs_
  std::map<std::string, std::string> addition_info_;
};
}

#endif  // INC_GRAPH_AXIS_TYPE_INFO_H_
