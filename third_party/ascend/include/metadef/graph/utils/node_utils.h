/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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

#ifndef INC_GRAPH_UTILS_NODE_UTILS_H_
#define INC_GRAPH_UTILS_NODE_UTILS_H_

#include <set>
#include <map>
#include <vector>
#include "external/graph/operator.h"
#include "external/graph/types.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "graph/compute_graph.h"

/*lint -e148*/
namespace ge {
// Op types of Const like Opps.
extern const std::set<std::string> kConstOpTypes;

// Op types of Enter like Opps.
extern const std::set<std::string> kEnterOpTypes;
// Op types of Merge like Opps.
extern const std::set<std::string> kMergeOpTypes;
// Op types of Switch like Opps.
extern const std::set<std::string> kSwitchOpTypes;
// Op types of NextIteration like Opps.
extern const std::set<std::string> kNextIterationOpTypes;
// Op types of Exit like Opps.
extern const std::set<std::string> kExitOpTypes;

// Op types of If like Opps.
extern const std::set<std::string> kIfOpTypes;
// Op types of While like Opps.
extern const std::set<std::string> kWhileOpTypes;
// Op types of Case like Opps.
extern const std::set<std::string> kCaseOpTypes;
// Op types of For like Opps.
extern const std::set<std::string> kForOpTypes;

class NodeUtils {
 public:
  static graphStatus ClearInDataAnchor(const NodePtr &node_ptr, const InDataAnchorPtr &in_data_anchor);
  static graphStatus SetAllAnchorStatus(const NodePtr &node_ptr);
  static graphStatus SetAllAnchorStatus(Node &node);
  static bool IsAnchorStatusSet(const NodePtr &node_ptr);
  static bool IsAnchorStatusSet(const Node &node);

  static graphStatus MoveOutputEdges(const NodePtr &origin_node, const NodePtr &new_node);

  static void UpdateIsInputConst(const NodePtr &node_ptr);
  static void UpdateIsInputConst(Node &node);
  static bool IsConst(const Node &node);
  static void UnlinkAll(const Node &node);

  static bool ClearInputDesc(const OpDescPtr &op_desc, const uint32_t index);
  static bool ClearOutputDesc(const OpDescPtr &op_desc, const uint32_t index);

  static graphStatus AppendInputAnchor(const NodePtr &node, const uint32_t num);
  static graphStatus RemoveInputAnchor(const NodePtr &node, const uint32_t num);

  static graphStatus AppendOutputAnchor(const NodePtr &node, const uint32_t num);
  static graphStatus RemoveOutputAnchor(const NodePtr &node, const uint32_t num);

  static GeTensorDesc GetOutputDesc(const Node &node, const uint32_t index);
  // check node whether unknown shape.If node shape contain -1 or -2,out param "is_unknow" will be true;
  // for func op, it will check subgraph yet, if some node shape of subgraph contain -1 or -2,
  // the out param "is_unknow" will be true too
  static graphStatus GetNodeUnknownShapeStatus(const Node &node, bool &is_unknow);

  static std::string GetNodeType(const Node &node);
  static std::string GetNodeType(const NodePtr &node);

  static graphStatus GetDirectSubgraphs(const NodePtr &node, std::vector<ComputeGraphPtr> &subgraphs);
  static ComputeGraphPtr GetSubgraph(const Node &node, const uint32_t index);
  static graphStatus SetSubgraph(Node &node, const uint32_t index, const ComputeGraphPtr &subgraph);
  static graphStatus AddSubgraph(Node &node, const std::string &subgraph_name, const ComputeGraphPtr &subgraph);
  static NodePtr CreatNodeWithoutGraph(const OpDescPtr op_desc);
  /// Check if node is input of subgraph
  /// @param [in] node
  /// @return bool
  static bool IsSubgraphInput(const NodePtr &node);
  static bool IsSubgraphInput(const Node *const node);

  /// Check if node is output of subgraph
  /// @param [in] node
  /// @return bool
  static bool IsSubgraphOutput(const NodePtr &node);

  /// @brief Get subgraph original input node.
  /// @param [in] node
  /// @return Node
  static NodePtr GetParentInput(const Node &node);
  static NodePtr GetParentInput(const NodePtr &node);
  /// @brief Get subgraph original input node and corresponding out_anchor.
  /// @param [in] node
  /// @return NodeToOutAnchor  node and out_anchor which linked to in_param node
  static NodeToOutAnchor GetParentInputAndAnchor(const NodePtr &node);
  /// @brief Get subgraph original input node and corresponding out_anchor corss subgraph.
  /// @param [in] node
  /// @return NodeToOutAnchor  node and out_anchor which linked to in_param node
  static NodeToOutAnchor GetParentInputAndAnchorCrossSubgraph(const NodePtr &node);

  /// @brief Get is dynamic shape graph from node.
  /// @param [in] node
  /// @return bool
  static bool IsDynamicShape(const Node &node);
  static bool IsDynamicShape(const NodePtr &node);

  /// @brief Check is varying_input for while node
  /// @param [in] node: Data node for subgraph
  /// @return bool
  static bool IsWhileVaryingInput(const ge::NodePtr &node);

  /// @brief Get subgraph input is constant.
  /// @param [in] node
  /// @param [out] string
  /// @return bool
  static bool GetConstOpType(const NodePtr &node, std::string &type);

  /// @brief Remove node-related subgraphs, including subgraphs of nodes in the subgraph.
  /// @param [in] node
  /// @return return GRAPH_SUCCESS if remove successfully, other for failed.
  static graphStatus RemoveSubgraphsOnNode(const NodePtr &node);

  /**
   * 获取`node`挂载的所有子图中的索引为`index`的Data节点集合;
   * 每个子图最多能找到一个跟`index`匹配的Data节点
   * @param node
   * @param index
   * @return
   */
  static std::vector<NodePtr> GetSubgraphDataNodesByIndex(const Node &node, const int32_t index);

  /**
 * 获取`node`挂载的所有子图中的NetOutput节点集合;
 * 每个子图有且只有一个NetOutput节点
 * @param node
 * @return
 */
  static std::vector<NodePtr> GetSubgraphOutputNodes(const Node &node);

  /**
 * 获取`node`所在的图对应的根图
 * @param node
 * @return
 */
  static ComputeGraphPtr FindRootGraph(const Node &node);

  /**
   * 根据`node_filter`获取被node控制的输出节点
   * @param node
   * @param node_filter 控制边拷贝白名单过滤器，可以通过传递此参数实现满足条件的输出节点的获取
   * @return
   */
  static std::vector<NodePtr> GetOutControlNodes(const Node &node, const NodeFilter &node_filter);

  /**
   * 根据`node_filter`获取控制node的输入节点
   * @param node
   * @param node_filter 控制边拷贝白名单过滤器，可以通过传递此参数实现满足条件的输入节点的获取
   * @return
   */
  static std::vector<NodePtr> GetInControlNodes(const Node &node, const NodeFilter &node_filter);

  static NodePtr GetInDataNodeByIndex(const Node &node, const int32_t index);
  static std::pair<NodePtr, OutDataAnchorPtr> GetInDataNodeAndAnchorByIndex(const Node &node, const int32_t index);

  static std::vector<std::pair<InDataAnchorPtr, NodePtr>> GetOutDataNodesWithAnchorByIndex(const Node &node,
                                                                                           const int32_t index);

  /**
  * 适用于`node`节点作为子图中的Data占位节点时，获取根图中父节点对应的实际输入节点的类型
  * 其他情况返回`node`本身的节点类型
  * @param node
  * @return
  */
  static std::string GetInConstNodeTypeCrossSubgraph(const ge::NodePtr &node);

  /**
* 适用于`node`节点作为子图中的Data占位节点时，获取根图中父节点对应的实际输入节点对象
* 其他情况返回`node`本身
* @param node
* @return
*/
  static NodePtr GetInNodeCrossSubgraph(const ge::NodePtr &node);

  /// @brief Get peer input node, supported get cross PartitionedCall .
  /// @param [in] node, current node
  /// @param [in] index, current node the index'th input, if it is PartionedCall's subgraph Data, please assign 0
  /// @param [out] peer_node,
  ///          A(PartionedCall_0)->B(PartionedCall_1)
  ///          PartionedCall_0's subgraph: Data->A->Netoutput
  ///          PartionedCall_1's subgraph: Data1->B->Netoutput
  ///          If it is called like GetInNodeCrossPartionCallNode(B,0,peer_node)or(Data1,0,peer_node), peer_node is A
  /// @param [out] peer_out_anchor_index, peer_node's corresponding out anchor's index
  /// @return [graphStatus] running result of this function
  static graphStatus GetInNodeCrossPartionedCallNode(const NodePtr &node, uint32_t index, NodePtr &peer_node);
  static graphStatus GetInNodeCrossPartionedCallNode(const NodePtr &node, uint32_t index, NodePtr &peer_node,
                                                     int32_t &peer_out_anchor_index);

  static graphStatus SetNodeParallelGroup(Node &node, const char_t *const group_name);

  static graphStatus UpdateInputOriginalShapeAndShape(const Node &node, const uint32_t index, const GeShape &shape);
  static graphStatus UpdateOutputOriginalShapeAndShape(const Node &node, const uint32_t index, const GeShape &shape);
  static bool IsDtResourceNode(const NodePtr &node);
  static bool IsLikeAtomicClean(const NodePtr &node);
  /**
   * 用于判断identity节点是否被用于控制先读后写顺序的，如果是的话，
   * 则图优化的时候不能无脑删除identity节点来提升性能
   * @param node_ptr
   * @return
   */
  static bool IsIdentityUsefulForRWControl(const NodePtr &node_ptr);
  /**
   * 尝试通过pld占位节点对应的实际const节点来获取权重
   * @param node_ptr placeholder的占位节点，常见于图拆分中间状态的图的输入节点类型
   * @param ge_tensor 权重的承载对象，成功获取时ge_tensor被设置为非空
   * @return 失败时代表内部流程错误，成功时不代表一定获取到了权重
   */
  static graphStatus TryGetWeightByPlaceHolderNode(const NodePtr &node_ptr, ConstGeTensorPtr &ge_tensor);
  /**
  * 尝试通过Data占位节点对应的实际const节点来获取权重
  * @param node_ptr Data占位节点，常见于子图的输入节点类型
  * @param ge_tensor 权重的承载对象，成功获取时ge_tensor被设置为非空
  * @return 失败时代表内部流程错误，成功时不代表一定获取到了权重
  */
  static graphStatus TryGetWeightByDataNode(const NodePtr &node_ptr, ConstGeTensorPtr &ge_tensor);
  /**
   * 判断`node`的名称是否是`name`
   * @param node
   * @param name
   * @return 如果是的话，返回true，否则 false
   */
  static bool IsNameEqual(const NodePtr &node, const ge::char_t *const name);
  /**
   * 判断`node`的类型是否是`type`
   * @param node
   * @param type
   * @return
   */
  static bool IsTypeEqual(const NodePtr &node, const ge::char_t *const type);
};

struct NodeCompareKey {
  bool operator()(const NodePtr &n0, const NodePtr &n1) const {
    if ((n0 == nullptr) || (n1 == nullptr)) {
      return false;
    }
    if (n0->GetName() == n1->GetName()) {
      const auto graph0 = n0->GetOwnerComputeGraph();
      const auto graph1 = n1->GetOwnerComputeGraph();
      if ((graph0 == nullptr) || (graph1 == nullptr)) {
        return false;
      }
      return (graph0->GetName() < graph1->GetName());
    }
    return (n0->GetName() < n1->GetName());
  }
};
using OrderedNodeSet = std::set<NodePtr, NodeCompareKey>;
}  // namespace ge
/*lint +e148*/
#endif  // INC_GRAPH_UTILS_NODE_UTILS_H_
