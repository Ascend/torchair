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
#ifndef INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_TURBO_H
#define INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_TURBO_H
#include <vector>
#include <map>
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/utils/anchor_utils.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "register/graph_optimizer/fusion_common/fusion_turbo_utils.h"

namespace fe {
enum TensorUptType {
  UPDATE_NONE = 0,
  UPDATE_THIS = 1,
  UPDATE_PEER,
};

struct WeightInfo {
  ge::GeShape shape;
  ge::GeShape ori_shape;
  ge::DataType datatype;
  ge::DataType ori_datatype;
  ge::Format format;
  ge::Format ori_format;
  uint8_t *data;
  int64_t shape_size;
  size_t total_data_size; // data_size * sizeof(datatype). !!!Could be zero!!!
  inline void CalcTotalDataSize() {
    if (shape.GetDimNum() == 0) {
      shape_size = 1;
    } else {
      shape_size = shape.GetShapeSize();
    }

    if ((shape_size > 0) && (datatype < data_type_size.size())) {
      total_data_size = (static_cast<size_t>(shape_size)) * data_type_size[datatype];
    } else {
      total_data_size = 0;
    }
  }

  WeightInfo(const ge::GeTensorDesc &tensor_desc,
             void *data_p);

  WeightInfo(const ge::NodePtr &node, const int32_t &index,
             void *data_p);

  WeightInfo(const ge::GeShape &shape_p, const ge::GeShape &ori_shape_p,
             const ge::DataType &datatype_p, const ge::DataType &ori_datatype_p,
             const ge::Format &format_p, const ge::Format &ori_format_p, void *data_p);

  WeightInfo(ge::GeShape &&shape_p, ge::GeShape &&ori_shape_p,
             const ge::DataType &datatype_p, const ge::DataType &ori_datatype_p,
             const ge::Format &format_p, const ge::Format &ori_format_p, void *data_p);

  WeightInfo(const ge::GeShape &shape_p, const ge::DataType &datatype_p,
             const ge::Format &format_p, void *data_p);

  WeightInfo(ge::GeShape &&shape_p, const ge::DataType &datatype_p,
             const ge::Format &format_p, void *data_p);
};

class FusionTurbo {
 public:
  explicit FusionTurbo(const ge::ComputeGraphPtr &graph);

  explicit FusionTurbo(ge::ComputeGraph &graph);

  ~FusionTurbo();

  static Status BreakInput(const ge::NodePtr &node,
                           const vector<int32_t> &input_index);

  static Status BreakOutput(const ge::NodePtr &node,
                            const vector<int32_t> &output_index);

  static Status BreakAllInput(const ge::NodePtr &node);

  static Status BreakAllOutput(const ge::NodePtr &node);

  Status RemoveNodeWithRelink(const ge::NodePtr &node, const std::initializer_list<int32_t> &io_map = {});

  Status RemoveNodeWithRelink(const ge::NodePtr &node, const std::vector<int32_t> &io_map = {});

  Status RemoveNodeOnly(const ge::NodePtr &node);

  /* If the node has no subsequent nodes, remove it.
   * If the node has subsequent nodes, just return.
   * Parameter include_control_nodes:
   * If only_care_data_nodes = true, then we will ignore the control outputs. */
  Status RemoveDanglingNode(const ge::NodePtr &node, const bool &only_care_data_nodes = false);
  
  Status RemoveMultiNodesOnly(const std::vector<ge::NodePtr> &nodes);

  ge::NodePtr UpdateConst(const ge::NodePtr &node, const int32_t &index, const WeightInfo &w_info) const;

  /* 1. If index is larger than or equalt to the input size of node, add a weight
   * tensor and node as the last input of node.
   * 2. If index is less than the input size of node and:
   *        2.1 If the peer node of this input index is nullptr, we add a const node
   *        as input and update tensor desc. ---> Call AddConstNode.
   *        2.2 If the peer node of this input index is Const, we substitute the data
   *        of current Const and update tensor desc. ---> Call UpdateConst
   *        2.3 If the peer node of this input is other type, we just skip it. */
  ge::NodePtr AddWeight(const ge::NodePtr &node, const int32_t &index, const WeightInfo &w_info) const;

  /* Add weight after one output of node. For example:
   *       NodeA----> NodeB
   *           \----> NodeC
   * After calling AddWeightAfter(NodeA, 0, w_info), the graph will be like:
   *       NewWeight----> NodeB
   *               \----> NodeC
   *       NodeA(will be dangling)
   * The rule is adding weight in front of every peer out node of NodeA.
   */
  ge::NodePtr AddWeightAfter(const ge::NodePtr &node, const int32_t &index, const WeightInfo &w_info) const;

  ge::NodePtr AddWeight(const ge::NodePtr &node, const string& tensor_name, const WeightInfo &w_info) const;

  /* Add a weight tensor and node as the last input of node. */
  ge::NodePtr AddWeight(const ge::NodePtr &node, const WeightInfo &w_info) const;

  std::vector<ge::NodePtr> AddWeights(const ge::NodePtr &node,
                                      const vector<WeightInfo> &w_infos) const;

  static ge::GeTensorPtr MutableWeight(const ge::NodePtr &node, int32_t index);

  ge::NodePtr AddNodeOnly(const string &op_name, const string &op_type) const;

  static ge::NodePtr AddNodeOnly(ge::ComputeGraph &graph, const string &op_name, const string &op_type);

  ge::NodePtr AddNodeOnly(const string &op_name, const string &op_type,
                          size_t dynamic_num) const;

  static ge::NodePtr AddNodeOnly(ge::ComputeGraph &graph, const string &op_name, const string &op_type,
                                 size_t dynamic_num);

  static Status TransferOutCtrlEdges(const std::vector<ge::NodePtr> &nodes,
                                     const ge::NodePtr &new_node);

  static Status TransferInCtrlEdges(const std::vector<ge::NodePtr> &nodes,
                                    const ge::NodePtr &new_node);

  ge::NodePtr InsertNodeBefore(const string &op_name, const string &op_type,
                               const ge::NodePtr &base_node, const int32_t &base_input_index,
                               const int32_t &input_index = 0,
                               const int32_t &output_index = 0) const;

  ge::NodePtr InsertNodeAfter(const string &op_name, const string &op_type,
                              const ge::NodePtr &base_node, const int32_t &base_output_index,
                              const int32_t &input_index = 0, const int32_t &output_index = 0) const;

  static Status LinkInput(Relations &input_relations,
                          const ge::NodePtr &dst_node,
                          const TensorUptType &update_tensor = UPDATE_THIS);

  static Status LinkOutput(Relations &output_relations,
                           const ge::NodePtr &src_node,
                           const TensorUptType &update_tensor = UPDATE_THIS);

  static ge::NodePtr GetPeerOutNode(const ge::NodePtr &node, const int32_t &this_node_input_index);

  static std::vector<ge::NodePtr> GetPeerInNodes(const ge::NodePtr &node, const int32_t &this_node_output_index);

  /* Check whether there is a path from [node1's] output [index1] to [node2].
   * The default value is -1 and -1 means any output is ok. */
  static bool CheckConnected(const ge::NodePtr &node1, const ge::NodePtr &node2,
                             const int32_t &index1 = -1);

  /* Default update input 0 of node. */
  Status UpdateInputByPeer(const ge::NodePtr &node, const int32_t &index,
                           const ge::NodePtr &peer_node, const int32_t &peer_index) const;

  Status UpdateOutputByPeer(const ge::NodePtr &node, const int32_t &index,
                            const ge::NodePtr &peer_node, const int32_t &peer_index) const;

  static bool IsUnknownShape(const ge::NodePtr &node, const int32_t &index, const bool &is_input = true);

  static bool IsUnknownOriShape(const ge::NodePtr &node, const int32_t &index, const bool &is_input = true);

  ge::NodePtr MultiInOne(const string &node_name, const string &node_type,
                         Relations &input_relations,
                         Relations &output_relations,
                         const std::vector<ge::NodePtr> &old_nodes = {},
                         const bool &remove_old = true);

  Status MultiInOne(const ge::NodePtr &new_node,
                    Relations &input_relations,
                    Relations &output_relations,
                    const std::vector<ge::NodePtr> &old_nodes = {},
                    const bool &remove_old = true);
  
  static bool HasControl(const ge::NodePtr &node);

  static bool HasInControl(const ge::NodePtr &node);

  static bool HasOutControl(const ge::NodePtr &node);

  static bool HasOutData(const ge::NodePtr &node);

  static Status MoveDataOutputUp(const ge::NodePtr &node, int32_t index);

  /* move node to pre node if pre node has subgraph
   * @param node  current need move node
   * @param index node move input index
   **/
  Status GraphNodeUpMigration(const ge::NodePtr &node, const int32_t index);

  /* move node to next node if next node has subgraph
   * @param node  current need move node
   * @param index node move output index
   **/
  Status GraphNodeDownMigration(const ge::NodePtr &node, const int32_t index);

  static NodeIndex GetPeerInFirstPair(const ge::NodePtr &node, int32_t index);

  static NodeIndex GetPeerOutPair(const ge::NodePtr &node, int32_t index);
 private:
  /* AddWeight will do either AddConstNode or UpdateConst. */
  ge::NodePtr AddConstNode(const ge::NodePtr &node, const int32_t &index,
                           const WeightInfo &w_info) const;

  ge::ComputeGraphPtr graph_;
};
}
#endif // INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_TURBO_H
