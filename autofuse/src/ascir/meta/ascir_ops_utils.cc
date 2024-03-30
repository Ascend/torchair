/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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
#include "ascir.h"
#include "node_utils_ex.h"
#include "op_desc_utils.h"

namespace ascir {
namespace cg {
void AddToGraphFollowOp(const ge::Operator &op, ge::Operator &new_op) {
  auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  if (node == nullptr) {
    throw std::invalid_argument(
        "The input node should be created by bg::Function, "
        "the input node may be created from ops::OpType");
  }
  auto graph = node->GetOwnerComputeGraph();
  auto new_node = graph->AddNode(ge::OpDescUtils::GetOpDescFromOperator(new_op));
  auto tmp_new_op = ge::OpDescUtils::CreateOperatorFromNode(new_node);
  std::swap(tmp_new_op, new_op);
}
}  // namespace cg
}  // namespace ascir