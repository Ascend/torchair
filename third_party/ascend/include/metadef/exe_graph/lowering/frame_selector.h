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

#ifndef METADEF_CXX_INC_EXE_GRAPH_LOWERING_FRAME_SELECTOR_H_
#define METADEF_CXX_INC_EXE_GRAPH_LOWERING_FRAME_SELECTOR_H_
#include <vector>
#include "value_holder.h"
#include "common/checker.h"
namespace gert {
namespace bg {
/**
 * frame选择器，通过frame选择器，可以将执行图逻辑生成到执行的frame上
 *
 * 当前frame选择器还无法提供OnInitFrame/OnDeInitFrame功能，因为ValueHolder跨图连接的能力仅支持从父图向子图连接，
 * 若出现了Init向Main图中节点连边的场景，当前无法处理。对于Init/DeInit需求，按需开发即可
 */
class FrameSelector {
 public:
  /**
   * 选择Init图，将builder中的逻辑生成到Init图上，并返回**Init节点**的输出ValueHolder。
   * 注意：如果builder中创建的输出ValueHolder具有guarder，那么此guarder节点会被自动移动到DeInit图中
   * @param builder 执行图构建函数
   * @return 成功时，将builder返回的ValueHolderPtr作为本函数的返回值；失败时，本函数返回空vector
   */
  static std::vector<ValueHolderPtr> OnInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder);
  static ge::graphStatus OnInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder,
                                    std::vector<ValueHolderPtr> &init_graph_outputs,
                                    std::vector<ValueHolderPtr> &init_node_outputs);
  /**
   * 选择Main图，将builder中的逻辑生成到Main图上。
   *
   * 需要注意的是，本函数仅保证当前将builder中的逻辑生成到Main图上，但不保证其始终在Main图上。
   * 在lowering构图完成后，在图优化阶段，CEM等优化可能将Main图上的Node移动到Init图中。
   *
   * @param builder 执行图构建函数
   * @return 成功时，将builder返回的ValueHolderPtr作为本函数的返回值；失败时，本函数返回空vector
   */
  static std::vector<ValueHolderPtr> OnMainRoot(const std::function<std::vector<ValueHolderPtr>()> &builder);
  static ge::graphStatus OnMainRoot(const std::function<std::vector<ValueHolderPtr>()> &builder,
                                    std::vector<ValueHolderPtr> &outputs);
  
  /**
   * 选择Main图，将builder中的逻辑生成到Main图上, 并且保证builder生成的节点在main图最后执行
   *
   * @param builder 执行图构建函数
   * @return 成功时，将builder返回的ValueHolderPtr作为本函数的返回值；失败时，本函数返回nullptr
   */
  static ValueHolderPtr OnMainRootLast(const std::function<bg::ValueHolderPtr()> &builder);
};

ValueHolderPtr HolderOnInit(const ValueHolderPtr &holder);
}  // namespace bg
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_LOWERING_FRAME_SELECTOR_H_