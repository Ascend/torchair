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
#ifndef INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_TURBO_UTILS_H
#define INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_TURBO_UTILS_H
#include <array>
#include "graph/utils/op_desc_utils.h"
#include "graph/debug/ge_log.h"

#define FUSION_TURBO_NOTNULL(val, ret)                       \
  do {                                                  \
    if ((val) == nullptr) {                             \
      GELOGD("Parameter[%s] must not be null.", #val); \
      return ret;                         \
    }                                                   \
  } while (0)

namespace fe {
enum Direction {
  CURRENT = 0, /* 表示NodeIndex指示的是当前节点的对应输入输出。 */
  /* 当连接输入的场景，PEER模式下会获取<node, index>的对端输出节点和对端index。 */
  /* 当连接输出的场景，PEER模式下会获取<node, index>的所有对端输入节点和所有对端index。 */
  PEER = 1,
  /* 当连接输入的场景，PEER_SINGLE模式下会获取<node, index>的对端输出节点和对端index。和PEER一致。 */
  /* 当连接输出的场景，PEER_SINGLE模式下会获取<node, index>的第一个对端输出节点和对端index。 */
  PEER_SINGLE = 2
};

struct NodeIndex {
  ge::NodePtr node;
  int32_t index;
  Direction direction = CURRENT;
  NodeIndex() {
    node = nullptr;
    index = -1;
  }
  NodeIndex(const ge::NodePtr &node_param, int32_t index_param) {
    node = node_param;
    index = index_param;
  }

  NodeIndex(const std::pair<ge::NodePtr, int32_t> &node_index_pair) {
    node = node_index_pair.first;
    index = node_index_pair.second;
  }

  NodeIndex(const ge::NodePtr &node_param, int32_t index_param, Direction direction_param) {
    node = node_param;
    index = index_param;
    direction = direction_param;
  }
};
using NodeIndices = std::vector<NodeIndex>;

using ThisIndex = int32_t;

class Relations {
 public:
  Relations();

  Relations(const std::initializer_list<NodeIndex> &peer_indices);

  explicit Relations(const std::map<ThisIndex, NodeIndices> &relations_param);

  explicit Relations(std::map<ThisIndex, NodeIndices> &&relations_param);

  Relations(const Relations &relations_param);

  Relations(Relations &&relations_param) noexcept;

  Relations(ThisIndex this_index, const NodeIndex &peer_index);

  Relations(ThisIndex this_index, const NodeIndices &peer_indices);

  Relations(ThisIndex this_index, NodeIndex &&peer_index);

  Relations(ThisIndex this_index, NodeIndices &&peer_indices);

  Relations(const std::initializer_list<std::pair<ThisIndex, NodeIndex>> &peer_indices);

  Relations(const std::initializer_list<std::pair<ThisIndex, std::initializer_list<NodeIndex>>> &peer_indices_vec);

  /****** Interface Add from here. ******/
  Relations& Add(ThisIndex this_index, const NodeIndex &peer_index);

  Relations& Add(ThisIndex this_index, const std::initializer_list<NodeIndex> &peer_indices);

  Relations& Add(ThisIndex this_index, const NodeIndices &peer_indices);

  Relations& Add(ThisIndex this_index, NodeIndex &&peer_index);

  Relations& Add(ThisIndex this_index, NodeIndices &&peer_indices);

  /* 由于NodeIndex当连接输入或输出是完全不一样的，我们需要根据原始relations计算作为
   * 输入和输出的真正的对端节点，所以要求必须通过接口来修改relations。 */
  Relations& UpdatePeerIndex(ThisIndex this_index, const NodeIndices &peer_indices);

  Relations& UpdatePeerIndex(ThisIndex this_index, NodeIndices &&peer_indices);

  Relations& UpdatePeerIndex(const std::map<ThisIndex, NodeIndices> &peer_indices);

  Relations& UpdatePeerIndex(std::map<ThisIndex, NodeIndices> &&peer_indices);

  const std::map<ThisIndex, NodeIndices>& GetRelations();

  const std::map<ThisIndex, NodeIndices>& GetInRelations();

  const std::map<ThisIndex, NodeIndices>& GetOutRelations();

  Relations& operator=(const Relations &relations_param);

  Relations& operator=(Relations &&relations_param) noexcept;
 private:
  NodeIndex GetPeerInFirstPair(ThisIndex relation_index, const ge::NodePtr &node, int32_t index);

  void AppendPeerInAllPairs(ThisIndex relation_index, const ge::NodePtr &node, int32_t index);

  void PreProcessOneNodeIndex(ThisIndex index, const NodeIndex &node_index);
  void PreProcessNodeIndices(ThisIndex index, const NodeIndices &node_indices);

  void PreProcess();
  /* 我们在添加ori_relations的时候就把两个方向的节点都计算好。 */
  std::map<ThisIndex, NodeIndices> in_relations;

  std::map<ThisIndex, NodeIndices> out_relations;

  /* 如果key是输出的index，那么vector里存放的就是对端输入的index；在单输出多引用场景，
   * 对端输入的index可能有多个。
   * 如果key是输入的index，那么vector里存放的就是对端输出的index。对端输出只会有一个。 */
  std::map<ThisIndex, NodeIndices> ori_relations;
};

extern const std::array<size_t, static_cast<size_t>(ge::DT_MAX + 1)> data_type_size;
class FusionTurboUtils {
 public:
  static NodeIndex GetPeerInFirstPair(const ge::NodePtr &node, int32_t index);
  static NodeIndex GetPeerOutPair(const ge::NodePtr &node, int32_t index);
  static ge::NodePtr GetConstInput(const ge::NodePtr &node, int32_t index);
};
}
#endif // INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_TURBO_UTILS_H
