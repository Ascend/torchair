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
#ifndef METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
#include "storage_shape.h"
#include "tensor.h"
#include "continuous_vector.h"
#include "extended_kernel_context.h"
#include "tiling_data.h"
#include "ge/ge_api_error_codes.h"

namespace fe {
class PlatFormInfos;
}  // namespace fe

namespace gert {
/**
 * tiling kernel的context
 */
class TilingContext : public ExtendedKernelContext {
 public:
  const void *GetCompileInfo() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    const auto av = GetInput(index);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<void *>();
  }
  /**
   * 获取CompileInfo
   * @tparam T CompileInfo的类型
   * @return CompileInfo的指针
   */
  template<typename T>
  const T *GetCompileInfo() const {
    return reinterpret_cast<const T *>(GetCompileInfo());
  }
  /**
   * 获取输入shape，输入shape中包含了原始shape与运行时shape
   * @param index 输入index
   * @return 输入shape指针，index非法时返回空指针
   */
  const StorageShape *GetInputShape(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
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
   * **注意：只有在`IMPL_OP`实现算子时， 将对应输入设置为数据依赖后，才可以调用此接口获取tensor，否则行为是未定义的。**
   * @param index 输入index
   * @return 输入tensor指针，index非法时返回空指针
   */
  const Tensor *GetInputTensor(const size_t index) const {
    return GetInputPointer<Tensor>(index);
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor指针
   * @param ir_index IR原型定义中的index
   * @return tensor指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Tensor *GetOptionalInputTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入Tensor指针
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return Tensor指针，index或relative_index非法时，返回空指针
   */
  const Tensor *GetDynamicInputTensor(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, relative_index);
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入shape指针，shape中包含了原始shape与运行时shape
   * @param ir_index IR原型定义中的index
   * @return shape指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const StorageShape *GetOptionalInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<StorageShape>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入shape指针，shape中包含了原始shape与运行时shape
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return shape指针，index或relative_index非法时，返回空指针
   */
  const StorageShape *GetDynamicInputShape(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<StorageShape>(ir_index, relative_index);
  }
  /**
   * 根据输出index，获取输出shape指针，shape中包含了原始shape与运行时shape
   * @param index 输出index
   * @return 输出shape指针，index非法时，返回空指针
   */
  const StorageShape *GetOutputShape(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetOutputsNum()) {
      return nullptr;
    }

    const size_t offset = compute_node_info->GetInputsNum();
    return GetInputPointer<StorageShape>(offset + index);
  }

  /*
   * outputs, tiling的outputs以如下顺序排列：
   * outputs[0]: tiling-key
   * outputs[1]: block-dim
   * outputs[2]: atomic-clean-flag
   * outputs[3]: tiling-data
   * outputs[4]: workspace sizes
   * outputs[5]: tiling condition
   */
  enum TilingOutputIndex : uint32_t {
    kOutputTilingKey,
    kOutputBlockDim,
    kOutputAtomicCleanFlag,
    kOutputTilingData,
    kOutputWorkspace,
    kOutputTilingCond,
    // add new output definitions here
    kOutputNum
  };

  /**
   * 设置tiling key
   * @param tiling_key tiling key
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetTilingKey(const uint64_t tiling_key) {
    const auto p = GetOutputPointer<uint64_t>(kOutputTilingKey);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = tiling_key;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取tiling key
   * @return tiling key，获取失败时
   */
  uint64_t GetTilingKey() const {
    const auto p = GetOutputPointer<uint64_t>(kOutputTilingKey);
    if (p == nullptr) {
      return std::numeric_limits<uint64_t>::max();
    }
    return *p;
  }
  /**
   * 设置block dim
   * @param block_dim block dim
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetBlockDim(const uint32_t block_dim) {
    const auto p = GetOutputPointer<uint32_t>(kOutputBlockDim);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = block_dim;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取block dim
   * @return block dim
   */
  uint32_t GetBlockDim() const {
    const auto p = GetOutputPointer<uint32_t>(kOutputBlockDim);
    if (p == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    return *p;
  }
  /**
   * 设置tiling cond
   * @param tiling_cond tiling condition
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetTilingCond(int32_t tiling_cond) {
    const auto p = GetOutputPointer<int32_t>(kOutputTilingCond);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = tiling_cond;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取tiling cond
   * @return tiling cond:有效的tiling_cond大于等于0，若该值无效返回-1
   */
  int32_t GetTilingCond() const {
    const auto p = GetOutputPointer<int32_t>(kOutputTilingCond);
    if (p == nullptr) {
      return -1;
    }
    return *p;
  }
  /**
   * 设置是否需要atomic clean
   * @param atomic true/false代表是否需要做atomic clean
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetNeedAtomic(const bool atomic) {
    const auto p = GetOutputPointer<bool>(kOutputAtomicCleanFlag);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = atomic;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取是否需要atomic clean
   * @return true/false
   */
  bool NeedAtomic() const {
    const auto p = GetOutputPointer<bool>(kOutputAtomicCleanFlag);
    if (p == nullptr) {
      return false;
    }
    return *p;
  }
  /**
   * 获取有类型的tiling data指针
   * @tparam T tiling data类型，sizeof(T)不可以大于编译结果中指定的最大tiling data长度
   * @return tiling data指针，失败时返回空指针
   */
  template<typename T>
  auto GetTilingData() -> T* {
    auto tiling_data = GetRawTilingData();
    if (tiling_data == nullptr) {
      return nullptr;
    }
    if (tiling_data->GetCapacity() < sizeof(T)) {
      return nullptr;
    }
    tiling_data->SetDataSize(sizeof(T));
    return static_cast<T *>(tiling_data->GetData());
  }
  /**
   * 获取无类型的tiling data码流
   * @return tiling data指针，失败时返回空指针
   */
  TilingData *GetRawTilingData() {
    return *GetOutputPointer<TilingData *>(kOutputTilingData);
  }
  /**
   * 获取workspace sizes指针
   * @param workspace_count workspace的个数，传入的workspace个数不可以超过编译时指定的最大workspace个数
   * @return workspace sizes指针
   */
  size_t *GetWorkspaceSizes(const size_t workspace_count) {
    const auto workspace = GetOutputPointer<TypedContinuousVector<size_t>>(kOutputWorkspace);
    if (workspace == nullptr) {
      return nullptr;
    }
    if (workspace->SetSize(workspace_count) != ge::SUCCESS) {
      return nullptr;
    }
    return workspace->MutableData();
  }
   /**
   * 获取 workspace 个数
   * @return workspace 个数
   */
  size_t GetWorkspaceNum() const {
    const auto workspace = GetOutputPointer<TypedContinuousVector<size_t>>(kOutputWorkspace);
    if (workspace == nullptr) {
      return 0U;
    }
    return workspace->GetSize();
  }
  /**
   * 获取 fe::PlatFormInfos 指针
   * @return fe::PlatFormInfos 指针
   */
  fe::PlatFormInfos *GetPlatformInfo() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    const auto av = GetInput(index + 1U);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<fe::PlatFormInfos *>();
  }
};
static_assert(std::is_standard_layout<TilingContext>::value, "The class TilingContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
