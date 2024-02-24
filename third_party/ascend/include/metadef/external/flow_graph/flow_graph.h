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

#ifndef INC_EXTERNAL_FLOW_GRAPH_FLOW_GRAPH_H_
#define INC_EXTERNAL_FLOW_GRAPH_FLOW_GRAPH_H_

#include <vector>
#include <cstdint>

#include "flow_attr.h"
#include "graph/graph.h"
#include "process_point.h"

namespace ge {
namespace dflow {
class FlowOperator : public ge::Operator {
public:
  ~FlowOperator() override;

protected:
  FlowOperator(const char *name, const char *type);
};

class FlowData : public FlowOperator {
public:
  FlowData(const char *name, int64_t index);
  ~FlowData() override;
};

class FlowNodeImpl;
class FlowNode : public FlowOperator {
public:
  FlowNode(const char *name, uint32_t input_num, uint32_t output_num);
  ~FlowNode() override;
  FlowNode &SetInput(uint32_t dst_index, const FlowOperator &src_op, uint32_t src_index = 0);
  FlowNode &MapInput(uint32_t node_input_index, const ProcessPoint &pp, uint32_t pp_input_index,
                     const std::vector<DataFlowInputAttr> &attrs = {});
  FlowNode &MapOutput(uint32_t node_output_index, const ProcessPoint &pp, uint32_t pp_output_index);
  FlowNode &AddPp(const ProcessPoint &pp);
  FlowNode &SetBalanceScatter();
  FlowNode &SetBalanceGather();

 private:
  std::shared_ptr<FlowNodeImpl> impl_;
};

class FlowGraphImpl;
using FlowGraphImplPtr = std::shared_ptr<FlowGraphImpl>;
class FlowGraph {
public:
  explicit FlowGraph(const char *name);
  ~FlowGraph();
  const ge::Graph &ToGeGraph() const;
  FlowGraph &SetInputs(const std::vector<FlowOperator> &inputs);
  FlowGraph &SetOutputs(const std::vector<FlowOperator> &outputs);
  FlowGraph &SetOutputs(const std::vector<std::pair<FlowOperator, std::vector<size_t>>> &output_indexes);
  const char *GetName() const;

  /**
   * @brief Set the Contains N Mapping Node object.
   * default is false, when contain a n-mapping node, you need call this method to set value to true.
   * n-mapping means one2n(one input split to multi outputs), n2one((multi inputs combine to one output)) or
   * n2n(multi inputs generate multi outputs) mapping.
   * @param contains_n_mapping_node whether contain n-mapping node.
   * @return FlowGraph& return current object
   */
  FlowGraph &SetContainsNMappingNode(bool contains_n_mapping_node);

  /**
   * @brief Set the inputs align attrs.
   *
   * @param align_max_cache_num align max cache num(a cache can save a set of inputs),
   *                            0 means align is not enable, max value is 1024.
   * @param align_timeout align timeout(ms), -1 means never timeout, and over 0 and less than 600 * 1000(10 minitues).
   * @param dropout_when_not_align whether dropout data when over max buf num or timeout, default value=false.
   * @return FlowGraph& return current object
   */
  FlowGraph &SetInputsAlignAttrs(uint32_t align_max_cache_num, int32_t align_timeout,
                                 bool dropout_when_not_align = false);

 private:
  FlowGraphImplPtr impl_;
};
}  // namespace dflow
}  // namespace ge

#endif  // INC_EXTERNAL_FLOW_GRAPH_FLOW_GRAPH_H_
