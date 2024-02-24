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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_COMPUTE_NODE_INFO_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_COMPUTE_NODE_INFO_H_
#include <type_traits>
#include <cstdint>
#include <cstddef>
#include "graph/types.h"
#include "storage_format.h"
#include "runtime_attrs.h"
namespace gert {
class AnchorInstanceInfo {
 public:
  AnchorInstanceInfo() {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  };
  AnchorInstanceInfo(const uint32_t instance_start, const uint32_t instantiation_num)
      : instance_start_(instance_start), instantiation_num_(instantiation_num) {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }

  /**
   * 获取本输入/输出实例化的个数
   * @return 实例化个数
   */
  size_t GetInstanceNum() const {
    return instantiation_num_;
  }

  /**
   * 获取本输入/输出首个实例化的Anchor的index
   * @return 首个实例化的Anchor的index
   */
  size_t GetInstanceStart() const {
    return instance_start_;
  }

  /**
   * 设置本输入/输出首个实例化Anchor的index
   * @param instance_start 首个实例化Anchor的index
   */
  void SetInstanceStart(const uint32_t instance_start) {
    instance_start_ = instance_start;
  }

  /**
   * 设置本输入/输出实例化的个数
   * @param instantiation_num 实例化的个数
   */
  void SetInstantiationNum(const uint32_t instantiation_num) {
    instantiation_num_ = instantiation_num;
  }

 private:
  uint32_t instance_start_;
  uint32_t instantiation_num_;
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<AnchorInstanceInfo>::value, "The class AnchorInstanceInfo must be a POD");

class CompileTimeTensorDesc {
 public:
  CompileTimeTensorDesc() {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }
  /**
   * 获取DataType
   * @return DataType
   */
  ge::DataType GetDataType() const {
    return data_type_;
  }
  /**
   * 获取format
   * @return format
   */
  const StorageFormat &GetFormat() const {
    return storage_format_;
  }
  /**
   * 获取原始format
   * @return 原始format
   */
  ge::Format GetOriginFormat() const {
    return storage_format_.GetOriginFormat();
  }
  /**
   * 获取运行时format
   * @return 运行时format
   */
  ge::Format GetStorageFormat() const {
    return storage_format_.GetStorageFormat();
  }
  /**
   * 获取原始format向运行时format转换时的补维规则
   * @return 补维规则
   */
  ExpandDimsType GetExpandDimsType() const {
    return storage_format_.GetExpandDimsType();
  }
  /**
   * 设置 data type
   * @param data_type data type
   */
  void SetDataType(const ge::DataType data_type) {
    data_type_ = data_type;
  }
  /**
   * 设置运行时format
   * @param format 运行时format
   */
  void SetStorageFormat(const ge::Format format) {
    storage_format_.SetStorageFormat(format);
  }
  /**
   * 设置原始format
   * @param format 原始format
   */
  void SetOriginFormat(const ge::Format format) {
    storage_format_.SetOriginFormat(format);
  }
  /**
   * 设置原始format向运行时format转换时的补维规则
   * @param expand_dims_type 补维规则
   */
  void SetExpandDimsType(const ExpandDimsType &expand_dims_type) {
    storage_format_.SetExpandDimsType(expand_dims_type);
  }

 private:
  ge::DataType data_type_;
  StorageFormat storage_format_;
  uint8_t reserved_[40]; // Reservd field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<CompileTimeTensorDesc>::value, "The class CompileTimeTensorDesc must be a POD");

class ComputeNodeInfo {
 public:
  /**
   * 获取计算节点的node type
   * @return node type
   */
  const ge::char_t *GetNodeType() const {
    return node_type_;
  }
  /**
   * 获取计算节点的node name
   * @return node name
   */
  const ge::char_t *GetNodeName() const {
    return node_name_;
  }
  /**
   * 获取算子IR原型定义中的输入个数
   * @return IR原型中定义的输入个数
   */
  size_t GetIrInputsNum() const {
    return ir_inputs_num_;
  }
  /**
   * 获取算子IR原型定义中的输出个数
   * @return IR原型中定义的输出个数
   */
  size_t GetIrOutputsNum() const {
    return ir_outputs_num_;
  }
  /**
   * 获取计算节点的输入个数
   * @return 计算节点的输入个数
   */
  size_t GetInputsNum() const {
    return inputs_num_;
  }
  /**
   * 获取计算节点的输出个数
   * @return 计算节点的输出个数
   */
  size_t GetOutputsNum() const {
    return outputs_num_;
  }
  /**
   * 根据IR原型中的输入index，获取对应的实例化信息
   * @param ir_index IR原型定义中的输入index
   * @return 输入的实例化信息
   */
  const AnchorInstanceInfo *GetInputInstanceInfo(const size_t ir_index) const {
    if (ir_index >= ir_inputs_num_) {
      return nullptr;
    }
    const auto inputs = reinterpret_cast<const AnchorInstanceInfo *>(&place_holder);
    return inputs + ir_index;
  }
  /**
  * 根据IR原型中的输出index，获取对应的实例化信息
  * @param ir_index IR原型定义中的输出index
  * @return 输出的实例化信息
  */
  const AnchorInstanceInfo *GetOutputInstanceInfo(const size_t ir_index) const {
    if (ir_index >= ir_outputs_num_) {
      return nullptr;
    }
    const auto outputs = reinterpret_cast<const AnchorInstanceInfo *>(
        reinterpret_cast<const uint8_t *>(&place_holder) +
        sizeof(AnchorInstanceInfo) * ir_inputs_num_ +
        sizeof(CompileTimeTensorDesc) * (inputs_num_ + outputs_num_) +
        runtime_attr_size_);
    return outputs + ir_index;
  }
  /**
   * 获取计算节点输入的Tensor描述，注意，编译时无法确定的shape信息不在Tensor描述中
   * @param index 计算节点的输入index
   * @return Tensor描述
   */
  const CompileTimeTensorDesc *GetInputTdInfo(const size_t index) const {
    if (index >= inputs_num_) {
      return nullptr;
    }
    const auto inputs = reinterpret_cast<const CompileTimeTensorDesc *>(
        reinterpret_cast<const uint8_t *>(&place_holder) + sizeof(AnchorInstanceInfo) * ir_inputs_num_);
    return inputs + index;
  }
  /**
   * 获取计算节点输出的Tensor描述，注意，编译时无法确定的shape信息不在Tensor描述中
   * @param index 计算节点的输出index
   * @return Tensor描述
   */
  const CompileTimeTensorDesc *GetOutputTdInfo(const size_t index) const {
    if (index >= outputs_num_) {
      return nullptr;
    }
    const auto outputs = reinterpret_cast<const CompileTimeTensorDesc *>(
        reinterpret_cast<const uint8_t *>(&place_holder) + sizeof(AnchorInstanceInfo) * ir_inputs_num_ +
        sizeof(CompileTimeTensorDesc) * inputs_num_);
    return outputs + index;
  }
  /**
   * 获取计算节点上的属性值，仅IR定义的属性值会被返回，其他属性值被丢弃
   * @return 所有IR原型定义过的属性值，属性值按照IR原型定义的顺序依次保存
   */
  const RuntimeAttrs *GetAttrs() const {
    return reinterpret_cast<const RuntimeAttrs *>(reinterpret_cast<const uint8_t *>(&place_holder) +
                                                  sizeof(AnchorInstanceInfo) * ir_inputs_num_ +
                                                  sizeof(CompileTimeTensorDesc) * (inputs_num_ + outputs_num_));
  }
  /**
   * 设置计算节点的node type
   * @param node_type 计算节点的node type
   */
  void SetNodeType(const ge::char_t *node_type) {
    node_type_ = node_type;
  }
  /**
   * 设置计算节点的node name
   * @param node_name 计算节点的node name
   */
  void SetNodeName(const ge::char_t *node_name) {
    node_name_ = node_name;
  }
  /**
   * 根据IR原型中的输入index，获取对应的实例化信息
   * @param ir_index IR原型定义中的输入index
   * @return 输入的实例化信息
   */
  AnchorInstanceInfo *MutableInputInstanceInfo(const size_t ir_index) const;
  /**
   * 根据IR原型中的输出index，获取对应的实例化信息
   * @param ir_index IR原型定义中的输出index
   * @return 输出的实例化信息
   */
  AnchorInstanceInfo *MutableOutputInstanceInfo(const size_t ir_index) const;
  /**
   * 获取计算节点输入的Tensor描述，注意，编译时无法确定的shape信息不在Tensor描述中
   * @param index 计算节点的输入index
   * @return Tensor描述
   */
  CompileTimeTensorDesc *MutableInputTdInfo(const size_t index) const;
  /**
   * 获取计算节点输出的Tensor描述，注意，编译时无法确定的shape信息不在Tensor描述中
   * @param index 计算节点的输出index
   * @return Tensor描述
   */
  CompileTimeTensorDesc *MutableOutputTdInfo(const size_t index) const;
  /**
   * 获取计算节点上的属性值，仅IR定义的属性值会被返回，其他属性值被丢弃
   * @return 所有IR原型定义过的属性值，属性值按照IR原型定义的顺序依次保存
   */
  RuntimeAttrs *MutableAttrs() const;
  static ge::graphStatus CalcSize(const size_t ir_inputs_num, const size_t inputs_num,
                                  const size_t outputs_num, size_t &total_size);
  void Init(const size_t ir_inputs_num, const size_t inputs_num, const size_t outputs_num,
            const ge::char_t *node_name, const ge::char_t *node_type);
  static ge::graphStatus CalcSize(const size_t ir_inputs_num, const size_t ir_outputs_num, const size_t inputs_num,
                                  const size_t outputs_num, size_t &total_size);
  void Init(const size_t ir_inputs_num, const size_t ir_outputs_num, const size_t inputs_num, const size_t outputs_num,
            const size_t attr_size, const ge::char_t *node_name, const ge::char_t *node_type);

  ComputeNodeInfo() = delete;
  ComputeNodeInfo(const ComputeNodeInfo &) = delete;
  ComputeNodeInfo(ComputeNodeInfo &&) = delete;
  ComputeNodeInfo &operator=(const ComputeNodeInfo &) = delete;
  ComputeNodeInfo &operator=(ComputeNodeInfo &&) = delete;

 private:
  const ge::char_t *node_type_;
  const ge::char_t *node_name_;
  size_t ir_inputs_num_;
  size_t inputs_num_;
  size_t outputs_num_;
  size_t ir_outputs_num_;
  size_t runtime_attr_size_;
  uint8_t reserved_[24];  // Reserved field, 24+8, do not directly use when only 8-byte left
  // following by inputs-AnchorInstanceInfo, inputs-outputs-CompileTimeTensorDesc, RuntimeAttrs,
  // outputs-AnchorInstanceInfo
  uint64_t place_holder;
};
static_assert(std::is_standard_layout<ComputeNodeInfo>::value, "The class ComputeNodeInfo must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_COMPUTE_NODE_INFO_H_
