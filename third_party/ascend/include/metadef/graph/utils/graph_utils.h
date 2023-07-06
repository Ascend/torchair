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

#ifndef INC_GRAPH_UTILS_GRAPH_UTILS_H_
#define INC_GRAPH_UTILS_GRAPH_UTILS_H_

#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/utils/anchor_utils.h"
#include "cycle_detector.h"

/**
 * 图dump接口，用于把`compute_graph`对象序列化到文件，默认落盘到当前路径;
 * 如果`compute_graph`挂载了子图对象，子图对象也尝试进行落盘
 * 图的落盘行为受`DUMP_GE_GRAPH`和`DUMP_GRAPH_LEVEL`和`DUMP_GRAPH_PATH`环境变量的控制
 * DUMP_GE_GRAPH含义说明：
 * 1-全量dump
 * 2-不含有权重等数据的基础版dump
 * 3-只显示节点关系的精简版dump
 * DUMP_GRAPH_LEVEL含义说明：
 * 1-dump所有的图
 * 2-dump除子图外的所有图
 * 3-dump最后阶段的生成图
 * 4-dump入口阶段的生成图
 * DUMP_GRAPH_PATH含义说明：
 * 控制图的落盘的路径
 * @param compute_graph
 * @param name 用于拼接文件的名称
 */
#define GE_DUMP(compute_graph, name)                                                                                   \
  do {                                                                                                                 \
    ge::GraphUtils::DumpGEGraph((compute_graph), (name));                                                              \
    ge::GraphUtils::DumpGEGraphToOnnx(*(compute_graph), (name));                                                       \
    uint64_t i = 0U;                                                                                                   \
    for (const auto &sub_graph_func : (compute_graph)->GetAllSubgraphs()) {                                            \
      const auto sub_graph_func_name = std::string(name) + std::string("_sub_graph_") + std::to_string(i++);           \
      ge::GraphUtils::DumpGEGraph(sub_graph_func, sub_graph_func_name);                                                \
      ge::GraphUtils::DumpGEGraphToOnnx(*sub_graph_func, sub_graph_func_name);                                         \
    }                                                                                                                  \
  } while (false)

namespace ge {
enum class DumpLevel { NO_DUMP = 0, DUMP_ALL = 1, DUMP_WITH_OUT_DATA = 2, DUMP_WITH_OUT_DESC = 3, DUMP_LEVEL_END = 4 };
enum class MemType { OUTPUT_MEM, WORKSPACE_MEM };

struct MemReuseInfo {
  NodePtr node;
  MemType mem_type;
  uint32_t index;
};

enum IOType { kIn, kOut };

class NodeIndexIO {
 public:
  NodeIndexIO(const NodePtr &node, const uint32_t index, const IOType io_type)
      : node_(node), index_(index), io_type_(io_type), node_ptr_(node.get()) {
    ToValue();
  }
  NodeIndexIO(const NodePtr &node, const int32_t index, const IOType io_type)
      : node_(node), index_(static_cast<uint32_t>(index)), io_type_(io_type), node_ptr_(node.get()) {
    ToValue();
  }
  NodeIndexIO(const NodePtr &node, const int64_t index, const IOType io_type)
      : node_(node), index_(static_cast<uint32_t>(index)), io_type_(io_type), node_ptr_(node.get()) {
    ToValue();
  }
  NodeIndexIO(const Node *node, const uint32_t index, const IOType io_type)
      : node_(nullptr), index_(index), io_type_(io_type), node_ptr_(node) {
    ToValue();
  }
  ~NodeIndexIO() {}

  const std::string &ToString() const {
    return value_;
  }

  void ToValue() {
    if (node_ptr_ != nullptr) {
      value_ = node_ptr_->GetName() + ((io_type_ == kOut) ? "_out_" : "_in_") + std::to_string(index_);
    }
  }

  NodePtr node_ = nullptr;
  uint32_t index_ = 0U;
  IOType io_type_ = kOut;
  std::string value_;
  const Node *node_ptr_ = nullptr;
};

class GraphUtils {
 public:
  /**
   * pipline拆分场景获取`compute_graph`的`PARTITIONEDCALL`子图
   * @param compute_graph
   * @param independent_compile_subgraphs:出参，pipline拆分场景返回子图对象，非拆分场景返回`compute_graph`本身
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus GetIndependentCompileGraphs(const ComputeGraphPtr &compute_graph,
                                                 std::vector<ComputeGraphPtr> &independent_compile_subgraphs);

  /**
   * `src`和`dst`进行连边，`dst`作为InDataAnchorPtr, 最多允许一个对端OutDataAnchorPtr
   * @param src
   * @param dst
   * @return 如果`dst`已经有数据输入，则返回GRAPH_FAILED
   */
  static graphStatus AddEdge(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst);

  static graphStatus AddEdge(const AnchorPtr &src, const AnchorPtr &dst);

  static graphStatus AddEdge(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst);

  static graphStatus AddEdge(const OutDataAnchorPtr &src, const InControlAnchorPtr &dst);

  /**
   * `src`和`dst`进行断边
   * @param src
   * @param dst
   * @return 如果`src`和`dst`没有连边关系，则返回GRAPH_FAILED
   */
  static graphStatus RemoveEdge(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst);

  static graphStatus RemoveEdge(const AnchorPtr &src, const AnchorPtr &dst);

  static graphStatus RemoveEdge(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst);

  static graphStatus RemoveEdge(const OutDataAnchorPtr &src, const InControlAnchorPtr &dst);

  /**
   * 替换`dst`的对端`src`为`new_src`
   * @param src
   * @param dst
   * @param new_src
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceEdgeSrc(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                    const OutDataAnchorPtr &new_src);
  /**
   * 替换`dst`的对端`src`为`new_src`
   * @param src
   * @param dst
   * @param new_src
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceEdgeSrc(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst,
                                    const OutControlAnchorPtr &new_src);
  /**
   * 替换`src`的对端`dst`为`new_dst`
   * @param src
   * @param dst
   * @param new_dst
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceEdgeDst(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                    const InDataAnchorPtr &new_dst);
  /**
   * 替换`src`的对端`dst`为`new_dst`
   * @param src
   * @param dst
   * @param new_dst
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceEdgeDst(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst,
                                    const InControlAnchorPtr &new_dst);
  /**
   * 在`src`所属的node对象和`dst`所属的node对象之间插入`new_node`, 行为等价于替换`src`的对端`dst`为`new_node`的第`0`个
   * InDataAnchor, 同时替换`dst`的对端`src`为`new_node`的第`0`个OutDataAnchor
   * @param src
   * @param dst
   * @param new_node
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus InsertNodeBetweenDataAnchors(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                                  const NodePtr &new_node);
  /**
   * 从`compute_graph`上删除`直接`或者`间接`父节点为remove_node的所有子图对象
   * @param compute_graph
   * @param remove_node
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus RemoveSubgraphRecursively(const ComputeGraphPtr &compute_graph, const NodePtr &remove_node);

  /**
   * 从`compute_graph`中删除算子类型为`node_type`的所有关系，包括子图关系，从属关系，作为`compute_graph`的输入，输出的关系；
   * 仅删除，不进行断边连边，不保证删除后节点前后的控制关系传递
   * @param compute_graph
   * @param node_type
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus RemoveNodesByTypeWithoutRelink(const ComputeGraphPtr &compute_graph, const std::string &node_type);

  /**
   * 从`compute_graph`中删除`node`对象的所有关系，包括子图关系，从属关系，作为`compute_graph`的输入，输出的关系；
   * 仅删除，不进行断边连边，不保证删除后节点前后的控制关系传递
   * @param compute_graph
   * @param node
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus RemoveNodeWithoutRelink(const ComputeGraphPtr &compute_graph, const NodePtr &node);

  /**
   * 从`compute_graph`中删除`nodes`对象们的所有关系，包括子图关系，从属关系，作为`compute_graph`的输入，输出的关系；
   * 仅删除，不进行断边连边，不保证删除后节点前后的控制关系传递
   * 此接口在图规模比较大的时候，比遍历`nodes`节点依次调用`RemoveNodeWithoutRelink`更高效一些
   * @param compute_graph
   * @param nodes
   * @return
   */
  static graphStatus RemoveNodesWithoutRelink(const ComputeGraphPtr &compute_graph,
                                              const std::unordered_set<NodePtr> &nodes);
  /**
   * ComputeGraph图对象的全量深拷贝接口
   * @param src_compute_graph 需要是根图对象
   * @param dst_compute_graph
   * @return
   */
  static graphStatus CopyComputeGraph(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph);

  /**
    * ComputeGraph图对象的深拷贝接口
    * @param src_compute_graph 需要是根图对象
    * @param node_filter 节点拷贝白名单过滤器，可以通过传递此参数实现满足条件的节点的复制，不传递时代表全量拷贝
    * @param graph_filter 子图拷贝白名单过滤器，可以通过传递此参数实现满足条件的子图的复制，不传递时代表全量拷贝
    * @param dst_compute_graph
    * @return
    */
  static graphStatus CopyComputeGraph(const ComputeGraphPtr &src_compute_graph, const NodeFilter &node_filter,
                                      const GraphFilter &graph_filter, ComputeGraphPtr &dst_compute_graph);

  /**
  * ComputeGraph图对象的深拷贝接口
  * @param src_compute_graph
  * @param node_filter 节点拷贝白名单过滤器，可以通过传递此参数实现满足条件的节点的复制，不传递时代表全量拷贝
  * @param graph_filter 子图拷贝白名单过滤器，可以通过传递此参数实现满足条件的子图的复制，不传递时代表全量拷贝
  * @param dst_compute_graph
  * @param node_old_2_new 新旧节点映射关系
  * @param op_desc_old_2_new 新旧节点描述信息的映射关系
  * @param depth 子图拷贝深度, 最大支持为10
  * @return
  */
  static graphStatus CopyComputeGraph(const ComputeGraphPtr &src_compute_graph, const NodeFilter &node_filter,
                                      const GraphFilter &graph_filter, ComputeGraphPtr &dst_compute_graph,
                                      std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                      std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new, const int32_t depth);

  /**
   * ComputeGraph图对象的深拷贝接口
   * @param src_compute_graph
   * @param dst_compute_graph
   * @param node_old_2_new 新旧节点映射关系
   * @param op_desc_old_2_new 新旧节点描述信息的映射关系
   * @param depth 子图拷贝深度, 最大支持为10
   * @return
   */
  static graphStatus CopyComputeGraph(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph,
                                      std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                      std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new, const int32_t depth);
  /**
   * 拷贝OpDesc对象，跟`CopyOpDesc`方法的区别是`CloneOpDesc`的拷贝内容精简一些
   * 注意：此接口性能较差，且拷贝内容存在丢失，为了考虑兼容目前保留此接口，推荐直接使用OpDesc
   * 的拷贝构造函数来实现拷贝，拷贝构造函数性能好且内容不存在丢失
   * @param org_op_desc
   * @return
   */
  static OpDescPtr CloneOpDesc(const ConstOpDescPtr &org_op_desc);
  /**
   * 拷贝OpDesc对象，跟`CloneOpDesc`方法的区别是`CopyOpDesc`的拷贝内容多一些，
   * 包括函数指针成员等
   * 注意：此接口性能较差，且拷贝内容存在丢失，为了考虑兼容目前保留此接口，推荐直接使用OpDesc
   * 的拷贝构造函数来实现拷贝，拷贝构造函数性能好且内容不存在丢失
   * @param org_op_desc
   * @return
   */
  static OpDescPtr CopyOpDesc(const ConstOpDescPtr &org_op_desc);
  /**
   * 接口行为是在数据`src`锚点所属的`src_node`节点和数据`dsts`锚点所属的`dst_node`节点们之间插入一个`insert_node`节点,
   * 默认是`insert_node`的`0`号数据输入锚点和`0`号输出数据锚点参与连边，`insert_node`插入之后, `src_node`和`insert_node`
   * 作为一个整体与原来的`src_node`具备等价的控制和数据关系
   * @param src 源数据输出锚点
   * @param dsts 源数据输出锚点连接的目的数据输入锚点，使用vector的原因是存在一个源锚点给到多个目的锚点的情况
   * @param insert_node 表示要插入的节点
   * @param input_index 表示插入节点的哪个数据输入锚点要跟src相连，如果不传递，默认取0
   * @param output_index 表示插入节点的哪个数据输出锚点要跟dsts依次相连，如果不传递，默认取0
   * @return 如果插入成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus InsertNodeAfter(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                                     const NodePtr &insert_node, const uint32_t input_index = 0U,
                                     const uint32_t output_index = 0U);

  /**
   * 接口行为是在数据`dst`锚点所属的`dst_node`节点和其对端`src_node`节点之间插入一个`insert_node`节点,
   * 默认是`insert_node`的`0`号数据输入锚点和`0`号数据输出数据锚点参与连边，`insert_node`插入之后,
   * `dst_node`和`insert_node`作为一个整体与原来的`dst_node`具备等价的控制和数据关系
   * @param dst 目的数据输入锚点
   * @param insert_node 表示要插入的节点
   * @param input_index 表示插入节点的哪个数据输入锚点要跟dst的对端src锚点相连，如果不传递，默认取0
   * @param output_index 表示插入节点的哪个数据输出锚点要跟dst相连，如果不传递，默认取0
   * @return 如果插入成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus InsertNodeBefore(const InDataAnchorPtr &dst, const NodePtr &insert_node,
                                      const uint32_t input_index = 0U, const uint32_t output_index = 0U);
  /**
   * 从`compute_graph`智能指针管理的图对象的包含的nodes列表中删除`node`节点，仅仅是删除节点，
   * 不包含对node的断边和重新连边等操作
   * @param compute_graph
   * @param node
   * @return 如果删除成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus RemoveJustNode(const ComputeGraphPtr compute_graph, const NodePtr &node);

  /**
   * 从`compute_graph`图对象的包含的nodes列表中删除`node`节点，仅仅是删除节点，不包含对`node`的断边和重新连边等操作
   * @param compute_graph
   * @param node
   * @return 如果删除成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus RemoveJustNode(ComputeGraph &compute_graph, const NodePtr &node);

  /**
   * 记录`original_nodes`的原始name到node上
   * @param original_nodes
   * @param node
   */
  static void RecordOriginalNames(const std::vector<ge::NodePtr> original_nodes, const ge::NodePtr &node);
  /**
   * 记录`names_tmp`中的字段到node上
   * @param original_nodes
   * @param node
   */
  static void RecordOriginalNames(std::vector<std::string> names_tmp, const ge::NodePtr &node);
  /**
   * 图dump接口，用于把`graph`对象序列化到文件，默认落盘到当前路径
   * @param graph
   * @param suffix 用于拼接文件的名称
   * @param is_always_dump 如果值为true，则接口行为不受环境变量约束
   * @param user_graph_name 用于指定落盘的文件名和文件路径
   */
  static void DumpGEGraph(const ge::ComputeGraphPtr &graph, const std::string &suffix,
                          const bool is_always_dump = false, const std::string &user_graph_name = "");
  /**
    * 图dump接口，用于把`graph`对象序列化到文件，落盘到`path`指定的路径
    * @param graph
    * @param path 指定落盘的路径
    * @param suffix 用于拼接文件的名称
    */
  static void DumpGEGrph(const ge::ComputeGraphPtr &graph, const std::string &path, const std::string &suffix);
  /**
   * 图dump接口，用于把`graph`对象序列化到文件，落盘到`file_path`
   * @param graph
   * @param file_path 路径+文件名
   * @param dump_level DUMP_GE_GRAPH环境变量以函数入参的表达
   * @return
   */
  static graphStatus DumpGEGraphByPath(const ge::ComputeGraphPtr &graph, const std::string &file_path,
                                       const ge::DumpLevel dump_level);
  static graphStatus DumpGEGraphByPath(const ge::ComputeGraphPtr &graph, const std::string &file_path,
                                       const int64_t dump_level);
  /**
   * 从`file`文件反序列化得到`compute_graph`的图对象
   * @param file
   * @param compute_graph
   * @return
   */
  static bool LoadGEGraph(const char_t *const file, ge::ComputeGraph &compute_graph);
  /**
   * 从`file`文件反序列化得到`compute_graph`的智能指针对象
   * @param file
   * @param compute_graph
   * @return
   */
  static bool LoadGEGraph(const char_t *const file, ge::ComputeGraphPtr &compute_graph);

  /**
   * 图dump接口，用于把`graph`对象按照onnx的格式序列化到文件，默认落盘到当前路径
   * @param compute_graph
   * @param suffix 用于拼接文件名
   */
  static void DumpGEGraphToOnnx(const ge::ComputeGraph &compute_graph, const std::string &suffix);
  /**
   * 图dump接口，用于把`graph`对象按照onnx的格式序列化到文件，默认落盘到`path`路径
   * @param compute_graph
   * @param path 路径名
   * @param suffix 拼接的文件名
   */
  static void DumpGrphToOnnx(const ge::ComputeGraph &compute_graph, const std::string &path, const std::string &suffix);

  static bool ReadProtoFromTextFile(const char_t *const file, google::protobuf::Message *const proto);

  static void WriteProtoToTextFile(const google::protobuf::Message &proto, const char_t *const real_path);

  static graphStatus AppendInputNode(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * 孤立`node`, 根据`io_map`完成node的输入输出数据边的重连；同时会添加必要的控制边保证`node`的所有输入节点
   * 均在`node`的输出节点之前执行
   * @param node
   * @param io_map 把第`io_map[i]`个输入的对端输出，连接到第`i`个输出的对端输入。因此`io_map`的元素个数应该与
   * `node`的输出锚点的个数相等，如果`io_map[i]`小于0，则仅断开第`i`个输出锚点到对端的所有连边
   * @return
   */
  static graphStatus IsolateNode(const NodePtr &node, const std::initializer_list<int32_t> &io_map);
  static graphStatus IsolateNode(const NodePtr &node, const std::vector<int32_t> &io_map);
  /**
   * `node`应该是单输入单输出的节点，接口行为等价于`IsolateNode(node, {0})`
   * @param node
   * @return
   */
  static graphStatus IsolateNodeOneIO(const NodePtr &node);

  /**
   * 此接口对数据关系的处理与`ReplaceNodeDataAnchors`的处理行为一致， 在此基础上，
   * 复制了`old_node`的所有控制关系到`new_node`上，这也是要注意的一点：
   * `数据`关系是`移动`操作，`控制`关系是`复制`操作
   * @param new_node
   * @param old_node
   * @param inputs_map 用于指导输入数据锚点的替换，注意元素个数不应该超过`new_node`的输入锚点总个数
   * @param outputs_map 用于指导输出锚点的替换，注意元素个数不应该超过`new_node`的输出锚点总个数
   * @return 如果替换成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceNodeAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                        const std::initializer_list<int32_t> inputs_map,
                                        const std::initializer_list<int32_t> outputs_map);

  /**
   * `ReplaceNodeAnchors`的重载接口
   * @param new_node
   * @param old_node
   * @param inputs_map
   * @param outputs_map
   * @return
   */
  static graphStatus ReplaceNodeAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                        const std::vector<int32_t> &inputs_map,
                                        const std::vector<int32_t> &outputs_map);

  /**
   * 接口行为是根据`inputs_map`和`outputs_map`把`old_node`上的数据关系`移动`到`new_node`上；具体操作是
   * 把`old_node`的第`inputs_map[i]`/`outputs_map[i]`个数据锚点的数据关系替换到`new_node`的第`i`个
   * 数据锚点上, `i`的取值范围是[0, `inputs_map`/`outputs_map`的元素个数）; 如果`inputs_map[i]`/`outputs_map[i]`
   * 的值小于0或者不在`old_node`的锚点索引范围之内，那么`new_node`的第`i`个数据锚点的数据关系保持原样
   * @param new_node
   * @param old_node
   * @param inputs_map 用于指导输入数据锚点的替换，注意元素个数不应该超过`new_node`的输入锚点总个数
   * @param outputs_map 用于指导输出锚点的替换，注意元素个数不应该超过`new_node`的输出锚点总个数
   * @return
   */
  static graphStatus ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                            const std::initializer_list<int32_t> inputs_map,
                                            const std::initializer_list<int32_t> outputs_map);

  static graphStatus ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                            const std::vector<int32_t> &inputs_map,
                                            const std::vector<int32_t> &outputs_map);

  /**
   * 拷贝`src_node`的控制输入边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return
   */
  static graphStatus CopyInCtrlEdges(const NodePtr &src_node, const NodePtr &dst_node);
  /**
   * 移动`src_node`的控制输入边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return
   */
  static graphStatus MoveInCtrlEdges(const NodePtr &src_node, const NodePtr &dst_node);
  /**
   * 拷贝`src_node`的控制输出边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return
   */
  static graphStatus CopyOutCtrlEdges(const NodePtr &src_node, const NodePtr &dst_node);
  /**
   * 移动`src_node`的控制输出边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return
   */
  static graphStatus MoveOutCtrlEdges(NodePtr &src_node, NodePtr &dst_node);

  /**
   * 查找`graph`的根图，如果当前图就是根图或者当前图没有父图，则返回当前图
   * @param graph
   * @return
   */
  static ComputeGraphPtr FindRootGraph(ComputeGraphPtr graph);
  /**
   * 浅拷贝`graph`，并不会同步拷贝`graph`的子图
   * @param graph
   * @param suffix 用于拼接克隆图中的节点名称
   * @param input_nodes 返回克隆图中的输入节点
   * @param output_nodes 返回克隆图中的输出节点
   * @return
   */
  static ComputeGraphPtr CloneGraph(const ComputeGraphPtr &graph, const std::string &suffix,
                                    std::vector<NodePtr> &input_nodes, std::vector<NodePtr> &output_nodes);
  /**
   * 拷贝`src_compute_graph`图上的attr属性到`dst_compute_graph`上，
   * 需要注意的是 如果`dst_compute_graph`图上已经存在了某些同名属性，则会跳过这些属性的值的拷贝
   * @param src_compute_graph
   * @param dst_compute_graph
   */
  static void InheritOriginalAttr(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph);
  /**
   * 拷贝`src_node`节点及其所有有效的输入输出tensor上的attr属性到`dst_desc`上
   * @param dst_desc 目的OpDesc对象
   * @param src_node 源Node对象
   * @return 拷贝成功返回GRAPH_SUCCESS， 拷贝失败返回GRAPH_FAILED
   */
  static graphStatus CopyTensorAttrs(const OpDescPtr &dst_desc, const NodePtr &src_node);

  /**
   * 获取当前图里面的所有的节点的输入输出tensor的复用关系
   * @param graph
   * @param symbol_to_anchors 使用同一块内存特征值的所有的tensor
   * @param anchor_to_symbol tensor和内存特征值的对应关系
   * @return
   */
  static graphStatus GetRefMapping(const ComputeGraphPtr &graph,
                                   std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                   std::map<std::string, std::string> &anchor_to_symbol);

  /// Determine if the graph is a UNKNOWN_SHAPE graph based on whether the graph and all subgraphs
  /// of the graph have UNKNOWN_SHAPE operators or not.
  /// Note: This function will only look 'down' from the graph, not 'up'. For example, the following
  /// scenario (K for known shape, U for unknown shape), ROOT graph is UNKNOWN_SHAPE while SUB graph is KNOWN_SHAPE
  /// ROOT graph:      A -----> B -----> C
  ///                  K    subgraph     U
  ///                           |
  ///                           V
  /// SUB graph:          D --> E --> F
  ///                     K     K     K
  /// @param [in] graph
  /// @return bool
  static bool IsUnknownShapeGraph(const ComputeGraphPtr &graph);

  static NodePtr FindNodeFromAllNodes(ComputeGraphPtr &graph, const std::string &name);
  static std::vector<NodePtr> FindNodesByTypeFromAllNodes(ComputeGraphPtr &graph, const std::string &type);
  /**
   * 判断当前`out_data_anchor`是否复用了输入anchor的内存
   * @param out_data_anchor
   * @param reuse_in_index 复用的输入anchor的index
   * @return 如果存在复用关系，返回true, 否则返回false
   */
  static bool IsRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index);
  /**
  * 针对含有`ATTR_NAME_NOPADDING_CONTINUOUS_INPUT`和`ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT`类型的节点
  * 单独封装的复用接口
  * @param out_data_anchor
  * @param reuse_in_index 出参，如果存在复用，值为0
  * @return 如果存在复用，返回true,负责返回false
  */
  static bool IsNoPaddingRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index);
  /**
   * 用于判断`node`是否直接或者间接从属于`graph`, `间接`的一种含义是`node`的父图是`graph`
   * @param graph
   * @param node
   * @return
   */
  static bool IsNodeInGraphRecursively(const ComputeGraphPtr &graph, const Node &node);

  /**
   * 获取所有`直接`父图为`graph`和`间接`父图为`graph`的子图对象合集
   * @param graph
   * @param subgraphs 子图对象的合集
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus GetSubgraphsRecursively(const ComputeGraphPtr &graph, std::vector<ComputeGraphPtr> &subgraphs);

  /**
   * 创建以`subgraph_name`拼接命名的子图对象`subgraph`，把`nodes`中的节点从`graph`中抽取出来放在`subgraph`中，
   * 完成图归属和节点连边关系的重建,`nodes`作为一个整体与`subgraph`父节点等价
   * @param graph
   * @param nodes
   * @param subgraph_name
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static ComputeGraphPtr BuildSubgraphWithNodes(const ComputeGraphPtr &graph, const std::set<NodePtr> &nodes,
                                                const std::string &subgraph_name);
  /**
   * `BuildSubgraphWithNodes`的重载接口
   * @param graph
   * @param nodes
   * @param subgraph_name
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static ComputeGraphPtr BuildSubgraphWithNodes(ComputeGraph &graph, const std::set<NodePtr> &nodes,
                                                const std::string &subgraph_name);
  /**
   * 作为`BuildSubgraphWithNodes`函数的逆向操作，会把`graph`展开其父图上，
   * 此接口支持子图的递归展开操作
   * @param graph 要展开的子图
   * @param filter 子图过滤器，用于过滤子图的子图是否要展开，不传递时不进行递归操作
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus UnfoldSubgraph(const ComputeGraphPtr &graph,
                                    const std::function<bool(const ComputeGraphPtr &)> &filter);
  /**
   * 作为`UnfoldSubgraph`的高阶版本，支持没有父子关系的图的展开操作，展开`graph`到`target_graph`上，
   * `graph`作为一个整体，等价替换掉`target_graph`中的`target_node`；
   * 此接口支持子图的递归展开操作
   * @param graph 要展开的子图
   * @param target_graph 展开到的目标图
   * @param target_node 子图要替换的目标节点
   * @param filter 图过滤器，用于过滤子图的子图是否要展开，不传递时不进行递归操作
   * @param depth
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus UnfoldGraph(const ComputeGraphPtr &graph, const ComputeGraphPtr &target_graph,
                                 const NodePtr &target_node, const function<bool(const ComputeGraphPtr &)> &filter,
                                 int32_t depth = 0);

  static CycleDetectorPtr CreateCycleDetector(const ComputeGraphPtr &graph);

  /**
     * 将node所有的输入、输出边断开，并移动到dst_graph
     * @param dst_graph 目的Graph，
     * @param node 需要移动的Node
     * @return 成功时，返回ge::GRAPH_SUCCESS
     */
  static graphStatus MoveNodeToGraph(const NodePtr &node, ComputeGraph &dst_graph);

 private:
  class GraphInfo {
   public:
    GraphInfo() = default;
    ~GraphInfo() = default;

   private:
    std::set<ge::NodePtr> nodes_;
    std::map<uint32_t, std::pair<ge::OutDataAnchorPtr, std::list<ge::InDataAnchorPtr>>> data_inputs_;
    std::map<uint32_t, std::pair<ge::OutDataAnchorPtr, std::list<ge::InDataAnchorPtr>>> data_outputs_;
    std::list<std::pair<ge::OutControlAnchorPtr, ge::InControlAnchorPtr>> ctrl_inputs_;
    std::list<std::pair<ge::OutControlAnchorPtr, ge::InControlAnchorPtr>> ctrl_outputs_;
    std::list<std::pair<ge::OutDataAnchorPtr, ge::InDataAnchorPtr>> inner_data_edges_;
    std::list<std::pair<ge::OutControlAnchorPtr, ge::InControlAnchorPtr>> inner_ctrl_edges_;
    friend class GraphUtils;
  };
  /// Get reference-mapping for in_data_anchors of node
  /// @param [in] node
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  static graphStatus HandleInAnchorMapping(const ComputeGraphPtr &graph, const NodePtr &node,
                                           std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                           std::map<std::string, std::string> &anchor_to_symbol);

  /// Get reference-mapping for out_data_anchors of node
  /// @param [in] node
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  static graphStatus HandleOutAnchorMapping(const NodePtr &node,
                                            std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                            std::map<std::string, std::string> &anchor_to_symbol);

  /// Handle input of subgraph
  /// @param [in] node
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  static graphStatus HandleSubgraphInput(const NodePtr &node,
                                         std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                         std::map<std::string, std::string> &anchor_to_symbol);

  /// Handle input of Merge op
  /// @param [in] node
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  static graphStatus HandleMergeInput(const NodePtr &node,
                                      std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                      std::map<std::string, std::string> &anchor_to_symbol);

  /// Handle output of subgraph
  /// @param [in] node
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  static graphStatus HandleSubgraphOutput(const NodePtr &node,
                                          std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                          std::map<std::string, std::string> &anchor_to_symbol);

  /// Relink all edges for cloned ComputeGraph.
  /// @param [in] node: original node.
  /// @param [in] suffix: node name suffix of new node.
  /// @param [in] all_nodes: all nodes in new graph.
  /// @return success: GRAPH_SUCESS
  static graphStatus RelinkGraphEdges(const NodePtr &node, const std::string &suffix,
                                      const std::unordered_map<std::string, NodePtr> &all_nodes);

  /// Union ref-mapping
  /// @param [in] exist_node_info1
  /// @param [in] exist_node_info2
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @param [out] symbol
  /// @return success: GRAPH_SUCESS
  static graphStatus UnionSymbolMapping(const NodeIndexIO &exist_node_info1, const NodeIndexIO &exist_node_info2,
                                        std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                        std::map<std::string, std::string> &anchor_to_symbol, std::string &symbol);

  /// Update symbol mapping with a new reference pair
  /// @param [in] cur_node_info
  /// @param [in] exist_node_info
  /// @param [out] symbol_to_anchors
  /// @param [out] anchor_to_symbol
  /// @return success: GRAPH_SUCESS
  static graphStatus UpdateRefMapping(const NodeIndexIO &cur_node_info, const NodeIndexIO &exist_node_info,
                                      std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                      std::map<std::string, std::string> &anchor_to_symbol);

  static void BuildGraphInfoFromNodes(const std::set<NodePtr> &nodes, GraphInfo &graph_info);

  static void BuildInDataEdgesFromNode(const NodePtr &node, const std::set<NodePtr> &nodes,
                                       std::map<OutDataAnchorPtr, size_t> &data_input_index_map, GraphInfo &graph_info);

  static NodePtr BuildSubgraphNode(ComputeGraph &graph, const std::string &graph_name, const GraphInfo &graph_info);

  static ComputeGraphPtr BuildSubgraph(const NodePtr &subgraph_node, const GraphInfo &graph_info,
                                       const std::string &subgraph_name);

  static graphStatus RelinkDataEdges(const NodePtr &subgraph_node, const GraphInfo &graph_info);

  static graphStatus RelinkCtrlEdges(const NodePtr &subgraph_node, const GraphInfo &graph_info);

  static graphStatus MergeInputNodes(const ComputeGraphPtr &graph, const NodePtr &target_node);

  static graphStatus MergeNetOutputNode(const ComputeGraphPtr &graph, const NodePtr &target_node);

  static bool MatchDumpStr(const std::string &suffix);

  static graphStatus CopyOpAndSubgraph(const ComputeGraphPtr &src_compute_graph, const NodeFilter &node_filter,
                                       const GraphFilter &graph_filter, ComputeGraphPtr &dst_compute_graph,
                                       std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                       std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                                       std::unordered_map<std::string, NodePtr> &all_new_nodes, const int32_t depth);

  static graphStatus CopyMembers(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph,
                                 const std::unordered_map<std::string, NodePtr> &all_new_nodes);

  static graphStatus CopyGraphImpl(const Graph &src_graph, Graph &dst_graph,
                                   const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                   const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new);
};

class ComputeGraphBuilder {
 public:
  ComputeGraphBuilder() : owner_graph_(nullptr) {}
  virtual ~ComputeGraphBuilder() = default;

  /// @brief Add node to graph
  /// @param [in] op_desc
  /// @return ComputeGraphBuilder
  virtual ComputeGraphBuilder &AddNode(const OpDescPtr &op_desc);

  /// @brief Add data-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] out_anchor_ind
  /// @param [in] dst_name
  /// @param [in] in_anchor_ind
  /// @return ComputeGraphBuilder
  virtual ComputeGraphBuilder &AddDataLink(const std::string &src_name, const uint32_t out_anchor_ind,
                                           const std::string &dst_name, const uint32_t in_anchor_ind);

  /// @brief Add ctrl-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] dst_name
  /// @return ComputeGraphBuilder
  virtual ComputeGraphBuilder &AddControlLink(const std::string &src_name, const std::string &dst_name);

  /// @brief Build graph
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return ComputeGraphPtr
  virtual ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) = 0;

  /// @brief Get node with name
  /// @param [in] name
  /// @return NodePtr
  NodePtr GetNode(const std::string &name);

  /// @brief Get all nodes
  /// @return std::vector<NodePtr>
  std::vector<NodePtr> GetAllNodes();

 protected:
  /// @brief Build nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildNodes(graphStatus &error_code, std::string &error_msg);

  /// @brief Build data-links
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildDataLinks(graphStatus &error_code, std::string &error_msg);

  /// @brief Build ctrl-links
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildCtrlLinks(graphStatus &error_code, std::string &error_msg);

 private:
  ComputeGraphBuilder(const ComputeGraphBuilder &) = delete;
  ComputeGraphBuilder &operator=(const ComputeGraphBuilder &) = delete;
  ComputeGraphBuilder(const ComputeGraphBuilder &&) = delete;
  ComputeGraphBuilder &operator=(const ComputeGraphBuilder &&) = delete;

  ComputeGraphPtr owner_graph_;
  // node_name -> node
  std::map<std::string, NodePtr> node_names_;
  std::vector<OpDescPtr> nodes_;
  // <src_node_name, out_anchor_ind> -> <dst_node_name, in_anchor_ind>
  std::vector<std::pair<std::pair<std::string, uint32_t>, std::pair<std::string, uint32_t>>> data_links_;
  // src_node_name -> dst_node_name
  std::vector<std::pair<std::string, std::string>> ctrl_links_;

  friend class CompleteGraphBuilder;
  friend class PartialGraphBuilder;
};

class CompleteGraphBuilder : public ComputeGraphBuilder {
 public:
  explicit CompleteGraphBuilder(const std::string name, const bool retval_flag = true)
      : ComputeGraphBuilder(), name_(name), parent_node_(nullptr), retval_flag_(retval_flag) {}
  CompleteGraphBuilder(const CompleteGraphBuilder &) = delete;
  CompleteGraphBuilder &operator=(const CompleteGraphBuilder &) = delete;
  CompleteGraphBuilder(const CompleteGraphBuilder &&) = delete;
  CompleteGraphBuilder &operator=(const CompleteGraphBuilder &&) = delete;
  ~CompleteGraphBuilder() = default;

  /// @brief Add node to graph
  /// @param [in] op_desc
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &AddNode(const OpDescPtr &op_desc) override;

  /// @brief Add data-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] out_anchor_ind
  /// @param [in] dst_name
  /// @param [in] in_anchor_ind
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &AddDataLink(const std::string &src_name, const uint32_t out_anchor_ind,
                                    const std::string &dst_name, const uint32_t in_anchor_ind) override;

  /// @brief Add ctrl-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] dst_name
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &AddControlLink(const std::string &src_name, const std::string &dst_name) override;

  /// @brief Set index_th input anchor for graph
  /// @param [in] index
  /// @param [in] node_names
  /// @param [in] anchor_inds
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &SetInput(const uint32_t index, const std::vector<std::string> &node_names,
                                 const std::vector<uint32_t> &anchor_inds);

  /// @brief Set index_th input of graph as useless
  /// @param [in] index
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &SetUselessInput(const uint32_t index);

  /// @brief Add output anchor for graph
  /// @param [in] owner_node_name
  /// @param [in] anchor_ind
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &AddOutput(const std::string &owner_node_name, uint32_t anchor_ind);

  /// @brief Add target for graph
  /// @param [in] target_name
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &AddTarget(const std::string &target_name);

  /// @brief Set parent-node of graph
  /// @param [in] parent_node
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &SetParentNode(const NodePtr &parent_node);

  /// @brief Set mapping-relation of parent-node in_anchor_ind & Data-node
  /// @param [in] input_mapping: index_of_graph_input -> in_anchor_index_of_parent_node
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &SetInputMapping(const std::map<uint32_t, uint32_t> &input_mapping);

  /// @brief Set mapping-relation of parent-node out_anchor_ind & NetOutput-node out_anchor_ind
  /// @param [in] output_mapping: index_of_graph_output -> out_anchor_index_of_parent_node
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &SetOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping);

  /// @brief Build graph
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return ComputeGraphPtr
  ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) override;

 private:
  /// @brief Add data nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void AddDataNodes(graphStatus &error_code, std::string &error_msg);

  /// @brief Add data node
  /// @param [in] index
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  NodePtr AddDataNode(const uint32_t index, graphStatus &error_code, std::string &error_msg);

  /// @brief Add RetVal nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void AddRetValNodes(graphStatus &error_code, std::string &error_msg);

  /// @brief Build target-nodes for graph
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildGraphTargets(graphStatus &error_code, std::string &error_msg);

  /// @brief Add NetOutput node
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void AddNetOutputNode(graphStatus &error_code, std::string &error_msg);

  /// @brief Build NetOutput nodes with data & ctrl edges
  /// @param [in] net_output_desc
  /// @param [in] peer_out_anchors
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildNetOutputNodeWithLink(const OpDescPtr &net_output_desc,
                                  const std::vector<OutDataAnchorPtr> &peer_out_anchors, graphStatus &error_code,
                                  std::string &error_msg);

  /// @brief process after build
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void PostProcess(graphStatus &error_code, std::string &error_msg);

  std::string name_;
  NodePtr parent_node_;
  bool retval_flag_;
  std::map<uint32_t, std::pair<std::vector<std::string>, std::vector<uint32_t>>> graph_inputs_;
  std::vector<std::pair<std::string, uint32_t>> graph_outputs_;
  std::vector<std::string> graph_targets_;

  // index_of_graph_input -> in_anchor_index_of_parent_node
  std::map<uint32_t, uint32_t> input_mapping_;
  // index_of_graph_output -> out_anchor_index_of_parent_node
  std::map<uint32_t, uint32_t> output_mapping_;
};

class PartialGraphBuilder : public ComputeGraphBuilder {
 public:
  PartialGraphBuilder() = default;
  PartialGraphBuilder(const PartialGraphBuilder &) = delete;
  PartialGraphBuilder &operator=(const PartialGraphBuilder &) = delete;
  PartialGraphBuilder(const PartialGraphBuilder &&) = delete;
  PartialGraphBuilder &operator=(const PartialGraphBuilder &&) = delete;
  ~PartialGraphBuilder() = default;

  /// @brief Add node to graph
  /// @param [in] op_desc
  /// @return PartialGraphBuilder
  PartialGraphBuilder &AddNode(const OpDescPtr &op_desc) override;

  /// @brief Add data-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] out_anchor_ind
  /// @param [in] dst_name
  /// @param [in] in_anchor_ind
  /// @return PartialGraphBuilder
  PartialGraphBuilder &AddDataLink(const std::string &src_name, const uint32_t out_anchor_ind,
                                   const std::string &dst_name, const uint32_t in_anchor_ind) override;

  /// @brief Add ctrl-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] dst_name
  /// @return PartialGraphBuilder
  PartialGraphBuilder &AddControlLink(const std::string &src_name, const std::string &dst_name) override;

  /// @brief Set owner graph
  /// @param [in] graph
  /// @return PartialGraphBuilder
  PartialGraphBuilder &SetOwnerGraph(const ComputeGraphPtr &graph);

  /// @brief Add exist node
  /// @param [in] node
  /// @return PartialGraphBuilder
  PartialGraphBuilder &AddExistNode(const NodePtr &exist_node);

  /// @brief Build multi nodes with links
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return ComputeGraphPtr
  ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) override;

 private:
  /// @brief Build exist nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildExistNodes(graphStatus &error_code, std::string &error_msg);

  std::vector<NodePtr> exist_nodes_;
};
}  // namespace ge

#endif  // INC_GRAPH_UTILS_GRAPH_UTILS_H_
