/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef GRAPH_CONNECTION_MATRIX_H_
#define GRAPH_CONNECTION_MATRIX_H_

#include "graph/node.h"
#include "graph/graph.h"
#include "graph/compute_graph.h"

namespace ge {
class ConnectionMatrixImpl;
using ConnectionMatrixImplPtr = std::shared_ptr<ConnectionMatrixImpl>;

class ConnectionMatrix {
public:
  explicit ConnectionMatrix(const ComputeGraphPtr &graph);
  ~ConnectionMatrix() = default;

  bool IsConnected(const NodePtr &a, const NodePtr &b) const;

  // inputs are all input nodes of parameter node.
  // if there is a path between A->B, then B will own A's
  // connectivity. The reason is ---
  // If some node can reach A, than it can also reach B.
  void SetConnectivity(const Node::Vistor<NodePtr> &inputs, const NodePtr &node);

  /* Computes the connectivity between two nodes in the
   * computation. The returned ConnectivityMatrix is constructed such that
   * ConnectivityMatrix::IsConnected(a, b) returns true iff there exists a
   * directed path (from producer to consumer) from 'a' to 'b'. Both data
   * connection and control connection are considered for connectivity.
   * A node is connected to itself. */
  graphStatus Generate(const ComputeGraphPtr &graph);

  // update reachablity map for fused nodes.
  void Update(const ComputeGraphPtr &graph, const std::vector<NodePtr> &fusion_nodes);

  void ExpandAndUpdate(const vector<ge::NodePtr> &fusion_nodes, const std::string &node_name);
private:
  ConnectionMatrixImplPtr impl_{nullptr};
};
}
#endif  // GRAPH_CONNECTION_MATRIX_H_
