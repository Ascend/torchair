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

#ifndef AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_EXE_GRAPH_ATTRS_H_
#define AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_EXE_GRAPH_ATTRS_H_
#include "graph/types.h"

namespace gert {
// 打在const节点上，表达const的值，
// todo 后面这个应该慢慢要废弃掉，通过buffer id代替
constexpr const ge::char_t *kConstValue = "value";

constexpr const ge::char_t *kGraph = "graph";

// 打在exe node上，代表本node执行的stage（例如DeInit）
constexpr const ge::char_t *kStage = "stage";

// 打在feed节点上（Data、InnerData），代表这是第几个输入
constexpr const ge::char_t *kFeedIndex = "index";

// 打在输出desc上，标识本输出不申请独立的内存，从某个node的某个输出index上引用过来使用
constexpr const ge::char_t *kRefFromNode = "RefFromNode";
constexpr const ge::char_t *kRefFromIndex = "RefFromIndex";

// 打在exe graph上，保存了本graph涉及的所有的ComputeNodeInfo
constexpr const ge::char_t *kComputeNodeInfo = "ComputeNodeInfo";

// 打在exe node上，用来标识本node所对应的计算图上的node的index
constexpr const ge::char_t *kComputeNodeIndex = "ComputeNodeIndex";

// 打在exe graph上，保存了本graph涉及的所有的KernelExtendInfo
constexpr const ge::char_t *kKernelExtendInfo = "KernelExtendInfo";

// 打在exe node上，用来标识本node所对应的kernel信息的index
constexpr const ge::char_t *kKernelExtendIndex = "KernelExtendInfoIndex";

// 打在exe node上，用来标识本node属于某一子图，且该子图内节点源自于子图外部，且对应子图外部节点拥有Guarder
constexpr const ge::char_t *kNodeWithGuarderOutside = "NodeWithGuarderOutside";

// 打在exe graph上，保存了本graph涉及的所有的二进制buffer（字符串、const值等）
constexpr const ge::char_t *kBuffer = "buffer";

// 打在exe graph上，保存了本graph涉及的ModelDesc信息
constexpr const ge::char_t *kModelDesc = "ModelDesc";

// 打在exe node上，类型是int，代表两层含义：1. 本node释放一个资源；2. 本node释放的资源位于本node的第n的输入index；n为属性的值
constexpr ge::char_t kReleaseResourceIndex[] = "ReleaseResourceIndex";

// 作为扩展属性打在exe graph上，类型是ge::ComputeGraphPtr，保存的是原来的计算图，未来会删除，因为无法做序列化，执行图序列化反序列化后会丢失该属性
constexpr const ge::char_t *kComputeGraph = "_compute_graph";

// 作为扩展属性打在exe node上，类型是PassChangedKernels，记录执行图经过pass后的新旧exe nodes输出的对应关系
constexpr const ge::char_t *kPassChangedInfo = "_pass_changed_info";

// 打在exe graph上，保存space_registry的智能指针
constexpr const ge::char_t *kSpaceRegistry = "SpaceRegistry";

// 打在exe graph上，保存外置权重文件目录的string
constexpr const ge::char_t *kExternalFileConstantDir = "ExternalFileConstantDir";
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_EXE_GRAPH_ATTRS_H_
