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

#ifndef COMMON_GRAPH_REF_RELATION_H_
#define COMMON_GRAPH_REF_RELATION_H_

#include <deque>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "graph/compute_graph.h"
#include "graph/ge_error_codes.h"
#include "node.h"

namespace ge {
enum InOutFlag {
  NODE_IN   = 0,  // input flag
  NODE_OUT  = 1,  // output flag
};

// RefCell的对象一经创建，便不允许去修改其数据成员。
struct RefCell {
  const std::string node_name;
  const ge::NodePtr node;
  const InOutFlag in_out;
  const int32_t in_out_idx;
  const std::string hash_key;

  explicit RefCell(const std::string &name, const ge::NodePtr &node_ptr, const InOutFlag in_out_flag, const int32_t idx)
      : node_name(name), node(node_ptr), in_out(in_out_flag), in_out_idx(idx),
        hash_key(std::string("")
                     .append(node_name)
                     .append(std::to_string(in_out))
                     .append(std::to_string(in_out_idx))
                     .append(std::to_string(PtrToValue(node.get())))) {}
  RefCell(const RefCell &ref_cell)
      : node_name(ref_cell.node_name), node(ref_cell.node), in_out(ref_cell.in_out), in_out_idx(ref_cell.in_out_idx),
        hash_key(ref_cell.hash_key) {}
  ge::RefCell &operator=(const ge::RefCell &ref_cell) = delete;
  bool operator == (const RefCell &c) const {
    return node_name == c.node_name && node == c.node && in_out == c.in_out && in_out_idx == c.in_out_idx;
  }
  ~RefCell() = default;
};

struct RefCellHash{
  size_t operator()(const RefCell &c) const {
    return std::hash<std::string>()(c.hash_key);
  }
};

class RefRelations {
 public:
  graphStatus LookUpRefRelations(const RefCell &key, std::unordered_set<RefCell, RefCellHash> &result);
  graphStatus BuildRefRelations(ge::ComputeGraph &graph);
  graphStatus Clear();

  RefRelations();
  ~RefRelations() = default;
 private:
  class Impl;
  std::shared_ptr<Impl> impl_ = nullptr;
};

}  // namespace ge
#endif  // COMMON_GRAPH_REF_RELATION_H_
