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
#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_DVPP_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_DVPP_CONTEXT_H_
#include <type_traits>
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/extended_kernel_context.h"

namespace gert {
/**
 * Dvpp kernel的context
 */
class DvppContext : public ExtendedKernelContext {
public:
  /**
   * 获取输入shape，输入shape中包含了原始shape与运行时shape
   * @param index 输入index
   * @return 输入shape指针，index非法时返回空指针
   */
  const StorageShape *GetInputShape(size_t index) const {
    auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetInputsNum()) {
      return nullptr;
    }

    return GetInputPointer<StorageShape>(index);
  }

  /**
   * 获取输入tensor
   *
   * **注意：只有在`IMPL_OP`实现算子时， 将对应输入设置为数据依赖后，
   * 才可以调用此接口获取tensor，否则行为是未定义的。**
   * @param index 输入index
   * @return 输入tensor指针，index非法时返回空指针
   */
  const Tensor *GetInputTensor(size_t index) const {
    return GetInputPointer<Tensor>(index);
  }

  /**
   * 根据输出index，获取输出shape指针，shape中包含了原始shape与运行时shape
   * @param index 输出index
   * @return 输出shape指针，index非法时，返回空指针
   */
  const StorageShape *GetOutputShape(size_t index) const {
    auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetOutputsNum()) {
      return nullptr;
    }

    size_t offset = compute_node_info->GetInputsNum();
    return GetInputPointer<StorageShape>(offset + index);
  }
};
static_assert(std::is_standard_layout<DvppContext>::value,
              "The class DvppContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_DVPP_CONTEXT_H_
