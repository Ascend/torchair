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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATOMICCLEANTILINGCONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATOMICCLEANTILINGCONTEXT_H_
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/continuous_vector.h"
namespace gert {
class AtomicCleanTilingContext : public TilingContext {
 public:
  /**
   * 获取workspace size的列表
   * @return workspace size列表
   */
  const ContinuousVector *GetCleanWorkspaceSizes() {
    return GetInputPointer<ContinuousVector>(0);
  }

  /**
   * 通过节点的输出index，获取需要清理的输出内存的大小
   * @param index 节点输出index
   * @return 需要清理的输出内存的大小
   */
  uint64_t GetCleanOutputSize(size_t index) {
    return GetInputValue<uint64_t>(index + 1U);
  }
};
static_assert(std::is_standard_layout<AtomicCleanTilingContext>::value,
              "The class AtomicCleanTilingContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATOMICCLEANTILINGCONTEXT_H_
