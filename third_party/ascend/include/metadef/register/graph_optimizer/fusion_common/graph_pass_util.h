/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_PASS_UTIL_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_PASS_UTIL_H_
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe {
enum class BackWardInheritMode {
  kInsertNode = 0,
  kFusedNode = 1,
  kInheritTrue = 2,
  kDoNotInherit = 3
};

using NodeTypeMap = std::unordered_map<std::string, std::map<std::string, ge::NodePtr>>;
using NodeTypeMapPtr = std::shared_ptr<NodeTypeMap>;
struct NodeMapInfo {
  int64_t run_count;
  NodeTypeMapPtr node_type_map;
};
using NodeMapInfoPtr = std::shared_ptr<NodeMapInfo>;
/** @brief define graph pass, which provides two interface: 1. run pass;
* 2. record op names before fusion */
class GraphPassUtil {
 public:
 using OriginOpAttrsVec = std::vector<std::vector<std::string>>;
 using UnorderedMapping = std::unordered_map<std::string, OriginOpAttrsVec>;
  /** set outputdesc attr for data dump
   *
   * @param origin_index,usually is origin node output index
   *
   * @param fusion_index,usually is fusion node output index
   *
   * @param origin_node, usually is origin node
   *
   * @param fusion_node, usually is fusion node
   */
  static void SetOutputDescAttr(const uint32_t &origin_index, const uint32_t &fusion_index,
                                const ge::NodePtr &origin_node, const ge::NodePtr &fusion_node);

  static void SetOutputDescAttr(ge::ConstGeTensorDescPtr &origin_tensor_desc, const int64_t origin_index,
                                const ge::OpDescPtr &origin_op_desc, const ge::GeTensorDescPtr &target_tensor_desc);

  /** get origin format for data dump
   *
   * @param tensor_desc,usually is output_desc
   *
   * @return format of this tensor_desc
   */
  static ge::Format GetDataDumpOriginFormat(const ge::GeTensorDescPtr &tensor_desc);

  static ge::Format GetDataDumpOriginFormat(ge::ConstGeTensorDescPtr &tensor_desc);

  /** set origin format for data dump
   *
   * @param origin format
   *
   * @param tensor_desc,usually is output_desc
   */
  static void SetDataDumpOriginFormat(const ge::Format &origin_format, const ge::GeTensorDescPtr &tensor_desc);

  /** set origin datatype for data dump
   *
   * @param origin datatype
   *
   * @param tensor_desc,usually is output_desc
   */
  static void SetDataDumpOriginDataType(const ge::DataType origin_data_type, const ge::GeTensorDescPtr &tensor_desc);

  /** get origin datatype for data dump
   *
   * @param tensor_desc,usually is output_desc
   *
   * @return format of this tensor_desc
   */
  static ge::DataType GetDataDumpOriginDataType(const ge::GeTensorDescPtr &tensor_desc);

  static ge::DataType GetDataDumpOriginDataType(ge::ConstGeTensorDescPtr &tensor_desc);

  static void AddNodeFromOpTypeMap(const NodeMapInfoPtr &node_map_info, const ge::NodePtr &node_ptr);

  static Status GetOpTypeMapToGraph(NodeMapInfoPtr &node_map_info, const ge::ComputeGraph &graph);

  static void RecordPassnameAndOriginalAttrs(const std::vector<ge::NodePtr> &original_nodes,
                                             std::vector<ge::NodePtr> &fus_nodes, const string &pass_name,
                                             const OriginOpAttrsVec &origin_op_attrs = OriginOpAttrsVec());

  static Status StoreAndUpdataOriginFusionPassName(const ge::OpDescPtr &op_desc,
                                                   const std::vector<ge::NodePtr> &original_nodes,
                                                   const std::string &pass_name);

  static void GetBackWardAttr(const std::vector<ge::NodePtr> &original_nodes,
                              bool &backward, BackWardInheritMode inherit_mode);

  static void InheritGraphRelatedAttr(const std::vector<ge::NodePtr> &original_nodes,
                                      const std::vector<ge::NodePtr> &fusion_nodes,
                                      BackWardInheritMode inherit_mode);

  /* If one of the original node has attribute like keep_dtype, the fused node
   * will inherit that attribute.
   * param inherit_mode: if fusion_nodes are newly inserted after original_nodes,
   * backward attr will only care about its farther nodes(pass farther nodes in
   * param original_nodes).
   * And if fusion_nodes are fused by a bunch of original_nodes, the backward attr
   * will not only care about original_nodes but also the input nodes of original_nodes. */
  static void InheritAttrFromOriNodes(const std::vector<ge::NodePtr> &original_nodes,
                                      const std::vector<ge::NodePtr> &fusion_nodes,
                                      BackWardInheritMode inherit_mode = BackWardInheritMode::kFusedNode);

  static void RecordOriginalOpAttrs(const std::vector<ge::NodePtr> &original_nodes,
                                    const ge::OpDescPtr &op_desc, const string &pass_name,
                                    const OriginOpAttrsVec &origin_op_attrs = OriginOpAttrsVec());

  static void RecordOriginalNames(const std::vector<ge::NodePtr> &original_nodes, const ge::NodePtr &node);

  static void AddNodeToNodeTypeMap(const NodeTypeMapPtr &node_type_map, const std::string &op_type,
                                   const ge::NodePtr &node_ptr);

  static void RemoveNodeFromNodeTypeMap(NodeTypeMapPtr &node_type_map, const std::string &op_type,
                                        const ge::NodePtr &node_ptr);

  static void GetNodesFromNodeTypeMap(NodeTypeMapPtr &node_type_map, const std::string &op_type,
                                      std::vector<ge::NodePtr> &nodes);
};

}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_PASS_UTIL_H_
