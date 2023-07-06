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
#ifndef GRAPH_CYCLE_DETECTOR_H_
#define GRAPH_CYCLE_DETECTOR_H_

#include "graph/node.h"
#include "graph/compute_graph.h"
#include "connection_matrix.h"

namespace ge {
class CycleDetector {
  friend class GraphUtils;
public:
  ~CycleDetector() = default;
  /* Detect whether there are cycles in graph
   * after fusing all nodes in param fusion_nodes.
   * Before call this func, you should call GenerateConnectionMatrix frist
   * to generate connection_matrix based on current graph.
   *
   * Compared with Cycle Detection
   * @param fusion_nodes: each vector in fusion_nodes
   * will be fused into an entity(which could contains
   * more than one node). The caller should put all original
   * nodes which are expected to be fused into one larger node
   * into each sub-vector of fusion_nodes.
   *
   * This function can tell whether there are a cycle after
   * fusing all nodes in fusion_nodes. Each vector in 2-d
   * vector fusion_nodes will be fused into an entity.
   *
   *
   * This interface cannot detect whether there are cycles
   * inside the fused nodes.
   *
   * e.g. {a, b, c, d} -> {e, f}
   * Because the edge information is not given for e and f
   * so this function we cannot tell if e and f are in a
   * cycle.
   * */
  bool HasDetectedCycle(const std::vector<std::vector<ge::NodePtr>> &fusion_nodes);

   /**
   * Update connection matrix based on graph.
   * Connection matrix is served for cycle detection.
   *
   * The first param graph, it should be the same one graph when contribue cycle_detector
   */
  void Update(const ComputeGraphPtr &graph, const std::vector<NodePtr> &fusion_nodes);

private:
  CycleDetector() = default;
  graphStatus Init(const ComputeGraphPtr &graph);
  std::unique_ptr<ConnectionMatrix> connectivity_{nullptr};
};

using CycleDetectorPtr = std::unique_ptr<CycleDetector>;
}
#endif  // GRAPH_CYCLE_DETECTOR_H_
