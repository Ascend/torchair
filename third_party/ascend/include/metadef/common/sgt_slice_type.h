/**
* Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef INC_COMMON_SGT_SLICE_TYPES_H_
#define INC_COMMON_SGT_SLICE_TYPES_H_

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace ffts {
const std::string kAttrSgtJsonInfo = "_sgt_json_info";
const std::string kAttrSgtStructInfo = "_sgt_struct_info";
const std::string kAttrSgtStructInfoDy = "_sgt_struct_info_dy";
const size_t kSgtTillingNum = 2U;

struct OpCut {
    int16_t split_cut_idx = -1;
    int16_t reduce_cut_idx = -1;
    int64_t cut_id = -1;
};

struct DimRange {
  int64_t lower;
  int64_t higher;
  bool operator==(const DimRange& dim_range) const {
    return (this->higher == dim_range.higher) && (this->lower == dim_range.lower);
  }
};

enum class AtomicType {
  None = 0,
  ADD = 1,
  SUB,
  MUL,
  DIV
};

struct ThreadSliceMap {
  uint32_t thread_scope_id;
  bool is_first_node_in_topo_order;
  uint32_t thread_mode;
  uint32_t node_num_in_thread_scope;
  bool is_input_node_of_thread_scope;
  bool is_output_node_of_thread_scope;
  std::vector<std::vector<std::vector<int64_t>>> ori_input_tensor_shape;
  std::vector<std::vector<std::vector<int64_t>>> ori_output_tensor_shape;
  std::string original_node;
  uint32_t slice_instance_num;
  uint32_t parallel_window_size;
  uint32_t thread_id;
  std::vector<std::vector<std::pair<std::string, uint32_t>>> dependencies;
  std::vector<uint32_t> core_num;
  std::vector<OpCut> cut_type;
  std::vector<AtomicType> atomic_types;
  std::vector<std::string> same_atomic_clean_nodes;
  std::vector<uint32_t> input_axis;
  std::vector<uint32_t> output_axis;
  std::vector<uint32_t> input_tensor_indexes;
  std::vector<uint32_t> output_tensor_indexes;
  std::vector<std::vector<std::vector<DimRange>>> input_tensor_slice;
  std::vector<std::vector<std::vector<DimRange>>> output_tensor_slice;
  std::vector<std::vector<std::vector<DimRange>>> ori_input_tensor_slice;
  std::vector<std::vector<std::vector<DimRange>>> ori_output_tensor_slice;
  std::vector<std::vector<int64_t>> input_cut_list;
  std::vector<std::vector<int64_t>> output_cut_list;
  ThreadSliceMap() : thread_scope_id(1U), is_first_node_in_topo_order(false), thread_mode(0U),
      node_num_in_thread_scope(1U), is_input_node_of_thread_scope(false), is_output_node_of_thread_scope(false),
      slice_instance_num(1U), parallel_window_size(1U), thread_id(0U) {}
  bool GetThreadMode() const {
    return (thread_mode == 0U) ? false : true;
  }
};

struct ThreadSliceMapDy {
    uint32_t slice_instance_num;
    uint32_t parallel_window_size;
    std::vector<uint32_t> input_tensor_indexes;
    std::vector<uint32_t> output_tensor_indexes;
    std::vector<std::vector<std::vector<int64_t>>> input_tensor_slice;
    std::vector<std::vector<std::vector<int64_t>>> output_tensor_slice;
    ThreadSliceMapDy() : slice_instance_num(1U), parallel_window_size(1U) {}
};

using ThreadSliceMapPtr = std::shared_ptr<ThreadSliceMap>;
using ThreadSliceMapDyPtr = std::shared_ptr<ThreadSliceMapDy>;
}  // namespace ffts
#endif  // INC_COMMON_SGT_SLICE_TYPES_H_
