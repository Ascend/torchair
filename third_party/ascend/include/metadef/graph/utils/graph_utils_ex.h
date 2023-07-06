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

#ifndef __INC_METADEF_GRAPH_UTILS_EX_H
#define __INC_METADEF_GRAPH_UTILS_EX_H

#include "graph/node.h"
#include "graph/compute_graph.h"
#include "external/graph/graph.h"

namespace ge {
class GraphUtilsEx {
 public:
  // Detach from ComputeGraph
  static graphStatus Verify(const ComputeGraphPtr &graph);
  static graphStatus InferOriginFormat(const ComputeGraphPtr &graph);
  static graphStatus InferShapeInNeed(const ComputeGraphPtr &graph);

  // Detach from GraphUtils
  static ComputeGraphPtr GetComputeGraph(const Graph &graph);
  static ComputeGraphPtr CreateGraphFromOperator(const std::string &name, const std::vector<Operator> &inputs);
  static Graph CreateGraphFromComputeGraph(const ComputeGraphPtr compute_graph);
  static GraphPtr CreateGraphPtrFromComputeGraph(const ComputeGraphPtr compute_graph);
  static void BreakConnect(const std::map<OperatorImplPtr, NodePtr> &all_nodes_infos);
  static graphStatus RecoverGraphOperators(const Graph &graph);
  static graphStatus CopyGraph(const Graph &src_graph, Graph &dst_graph);

 private:
  static graphStatus CopyGraphImpl(const Graph &src_graph, Graph &dst_graph,
                                   const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                   const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new);
};
} // namespace ge
#endif // __INC_METADEF_GRAPH_UTILS_EX_H
