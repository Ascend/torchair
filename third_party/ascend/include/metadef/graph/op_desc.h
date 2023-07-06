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

#ifndef INC_GRAPH_OP_DESC_H_
#define INC_GRAPH_OP_DESC_H_

#include <unordered_set>
#include "detail/attributes_holder.h"
#include "graph/range_vistor.h"
#include "graph/ge_tensor.h"

namespace ge {
using std::map;
using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;

class Operator;
class OpDescImpl;
class IRMetaData;
using OpDescImplPtr = std::shared_ptr<OpDescImpl>;

enum SubgraphType {
  kStatic,
  kDynamic,
  kSubgraphTypeEnd
};

enum IrInputType {
  kIrInputRequired,
  kIrInputOptional,
  kIrInputDynamic,
  kIrInputTypeEnd
};

enum IrOutputType {
  kIrOutputRequired,
  kIrOutputDynamic,
  kIrOutputTypeEnd
};

class OpDesc : public std::enable_shared_from_this<OpDesc>, public AttrHolder {
 public:
  using OpDescPtr = std::shared_ptr<OpDesc>;
  using ConstOpDescPtr = std::shared_ptr<const OpDesc>;

  template <class T>
  using Vistor = RangeVistor<T, shared_ptr<const OpDesc>>;

  friend class GraphBuilderImpl;

  friend class OperatorImpl;

  OpDesc(const std::string &name, const std::string &type);

  OpDesc(const OpDesc &op_desc);

  OpDesc(OpDesc &&op_desc);

  explicit OpDesc(const ge::proto::OpDef &op_def);

  OpDesc();

  ~OpDesc() override;

  bool operator==(const OpDesc &r_op_desc) const;

  std::string GetName() const;
  const char *GetNamePtr() const;

  void SetName(const std::string &name);

  std::string GetType() const;
  const char *GetTypePtr() const;

  void SetType(const std::string &type);

  void SetIrRelated(const OpDescPtr &op_desc);

  graphStatus AddInputDesc(const GeTensorDesc &input_desc);

  graphStatus AddInputDesc(const std::string &name, const GeTensorDesc &input_desc);

  graphStatus AddInputDesc(const uint32_t index, const ge::GeTensorDesc &input_desc);

  graphStatus AddInputDescMiddle(const std::string &name, const uint32_t num, const size_t index);

  graphStatus AddOutputDescMiddle(const std::string &name, const uint32_t num, const size_t index);

  graphStatus AddOutputDescForward(const std::string &name, const uint32_t num);

  graphStatus AddOptionalInputDesc(const std::string &name, const GeTensorDesc &input_desc);

  graphStatus UpdateInputDesc(const uint32_t index, const GeTensorDesc &tensor_desc);

  graphStatus UpdateInputDesc(const std::string &name, const GeTensorDesc &tensor_desc);

  bool InputIsSet(const std::string &name) const;

  const GeTensorDesc &GetInputDesc(const uint32_t index) const;

  const GeTensorDesc &GetInputDesc(const std::string &name) const;

  bool IsOptionalInput(const uint32_t index) const;

  Vistor<string> GetAllInputNames() const;

  GeTensorDescPtr MutableInputDesc(const uint32_t index) const;

  GeTensorDescPtr MutableInputDesc(const std::string &name) const;
  /**
   * 获取OpDesc的所有输入的GeTensorDesc对象的拷贝，
   * 需要注意拷贝行为对性能的影响
   * @return
   */
  Vistor<GeTensorDesc> GetAllInputsDesc() const;
  /**
   * 获取OpDesc的所有输入的GeTensorDesc对象的引用，
   * 无特殊需求，推荐使用此接口替代GetAllInputsDesc（）
   * @return
   */
  Vistor<GeTensorDescPtr> GetAllInputsDescPtr() const;

  size_t GetInputsSize() const;

  size_t GetAllInputsSize() const;

  graphStatus AddOutputDesc(const GeTensorDesc &output_desc);

  graphStatus AddOutputDesc(const std::string &name, const GeTensorDesc &output_desc);

  graphStatus UpdateOutputDesc(const uint32_t index, const GeTensorDesc &tensor_desc);

  graphStatus UpdateOutputDesc(const std::string &name, const GeTensorDesc &tensor_desc);

  const GeTensorDesc &GetOutputDesc(const uint32_t index) const;

  const GeTensorDesc &GetOutputDesc(const std::string &name) const;

  GeTensorDescPtr MutableOutputDesc(const uint32_t index) const;

  GeTensorDescPtr MutableOutputDesc(const std::string &name) const;

  uint32_t GetAllOutputsDescSize() const;

  Vistor<GeTensorDesc> GetAllOutputsDesc() const;

  Vistor<GeTensorDescPtr> GetAllOutputsDescPtr() const;

  size_t GetOutputsSize() const;

  ConstGeTensorDescPtr GetOutputDescPtr(const uint32_t index) const;

  ConstGeTensorDescPtr GetInputDescPtr(const uint32_t index) const;

  ConstGeTensorDescPtr GetInputDescPtrDfault(const uint32_t index) const;

  ConstGeTensorDescPtr GetInputDescPtr(const std::string &name) const;

  graphStatus AddDynamicInputDesc(const std::string &name, const uint32_t num, const bool is_push_back = true);

  graphStatus AddDynamicInputDescByIndex(const std::string &name, const uint32_t num, const size_t index);

  graphStatus AddDynamicOutputDesc(const std::string &name, const uint32_t num, const bool is_push_back = true);

  bool IsOptionalInput(const std::string &name) const;

  std::map<std::string, uint32_t> GetAllInputName() const;

  std::map<std::string, uint32_t> GetAllOutputName();

  std::map<std::string, uint32_t>& MutableAllInputName();

  std::map<std::string, uint32_t>& MutableAllOutputName();

  bool UpdateInputName(const std::map<std::string, uint32_t> input_name_idx);

  bool UpdateOutputName(const std::map<std::string, uint32_t> output_name_idx);

  void *GetTilingFuncInfo() const;

  void SetTilingFuncInfo(void *tiling_func_info);

  void *GetAtomicTilingFuncInfo() const;

  void SetAtomicTilingFuncInfo(void *atomic_tiling_func_info);

  graphStatus VerifyIR();

  graphStatus DefaultInferDataType();

  void AddInferFunc(const std::function<graphStatus(Operator &)> &func);
  void AddInferFormatFunc(const std::function<graphStatus(Operator &)> &func);
  void AddInferValueRangeFunc(const std::function<graphStatus(Operator &)> &func);
  void AddVerifierFunc(const std::function<graphStatus(Operator &)> &func);
  void AddInferDataSliceFunc(const std::function<graphStatus(Operator &)> &func);

  graphStatus DefaultInferFormat();

  std::function<graphStatus(Operator &)> GetInferFunc() const;
  std::function<graphStatus(Operator &)> GetVerifyFunc() const;
  std::function<graphStatus(Operator &)> GetInferFormatFunc() const;
  std::function<graphStatus(Operator &)> GetInferDataSliceFunc() const;
  std::function<graphStatus(Operator &)> GetInferValueRangeFunc() const;

  graphStatus CommonVerify() const;

  graphStatus AddRegisterInputName(const std::string &name);

  graphStatus AddRegisterOutputName(const std::string &name);

  std::vector<std::string> GetRegisterInputName() const;

  std::vector<std::string> GetRegisterOutputName() const;

  void AppendIrAttrName(const std::string &name);
  const std::vector<std::string> &GetIrAttrNames() const;

  void AppendIrInput(std::string name, IrInputType input_type);
  const std::vector<std::pair<std::string, IrInputType>> &GetIrInputs() const;
  size_t GetIrInputsSize() const;

  void AppendIrOutput(std::string name, IrOutputType output_type);
  const std::vector<std::pair<std::string, IrOutputType>> &GetIrOutputs() const;

  void RegisterDataTypeSymbol(const std::string &datatype_symbol, const TensorType &type_range);
  void RegisterDataTypeSymbol(const std::string &datatype_symbol, const ListTensorType &type_range);
  void RegisterIrInputDataTypeSymbol(const std::string &input_name, const std::string &datatype_symbol);
  void RegisterIrOutputDataTypeSymbol(const std::string &output_name, const std::string &datatype_symbol);

  using AttrHolder::AddRequiredAttr;
  using AttrHolder::DelAttr;
  using AttrHolder::GetAllAttrNames;
  using AttrHolder::GetAllAttrs;
  using AttrHolder::GetAttr;
  using AttrHolder::HasAttr;
  using AttrHolder::SetAttr;

  void SetId(const int64_t id);
  int64_t GetId() const;
  void SetStreamId(const int64_t stream_id);
  int64_t GetStreamId() const;
  void SetInputName(const std::vector<std::string> &input_name);
  std::vector<std::string> GetInputName() const;
  void SetSrcName(const std::vector<std::string> &src_name);
  std::vector<std::string> GetSrcName() const;
  void SetSrcIndex(const std::vector<int64_t> &src_index);
  std::vector<int64_t> GetSrcIndex() const;
  void SetInputOffset(const std::vector<int64_t> &input);
  std::vector<int64_t> GetInputOffset() const;
  void SetOutputOffset(const std::vector<int64_t> &output);
  std::vector<int64_t> GetOutputOffset() const;
  void SetDstName(const std::vector<std::string> &dst_name);
  std::vector<std::string> GetDstName() const;
  void SetDstIndex(const std::vector<int64_t> &dst_index);
  void SetWorkspace(const std::vector<int64_t> &workspace);
  std::vector<int64_t> GetWorkspace() const;
  void SetWorkspaceBytes(const std::vector<int64_t> &workspace_bytes);
  std::vector<int64_t> GetWorkspaceBytes() const;
  void SetIsInputConst(const std::vector<bool> &is_input_const);
  std::vector<bool> GetIsInputConst() const;

  void SetOpInferDepends(const std::vector<std::string> &depend_names);
  std::vector<std::string> GetOpInferDepends() const;

  std::string GetInputNameByIndex(const uint32_t index) const;
  std::string GetValidInputNameByIndex(const uint32_t index) const;
  int32_t GetInputIndexByName(const std::string &name) const;

  std::string GetOutputNameByIndex(const uint32_t index) const;

  int32_t GetOutputIndexByName(const std::string &name) const;

  void SetOpKernelLibName(const std::string &name);

  std::string GetOpKernelLibName() const;

  void SetOpEngineName(const std::string &name);

  std::string GetOpEngineName() const;

  void RegisterSubgraphIrName(const std::string &name, const SubgraphType type);
  const std::map<std::string, SubgraphType> &GetSubgraphIrNames() const;
  SubgraphType GetSubgraphTypeByIrName(const std::string &name) const;

  graphStatus AddSubgraphName(const std::string &name);
  const std::map<std::string, uint32_t> &GetSubgraphNameIndexes() const;

  std::string GetSubgraphInstanceName(const uint32_t index) const;
  const std::vector<std::string> &GetSubgraphInstanceNames() const;
  /// Does not provide functions `AddSubgraphInstance` or `AppendSubgraphInstance`,
  /// because this kind of functions will only append a new subgraph instance name
  /// at the tail of `subgraph_instance_names_` and ignore the synchronous change of `subgraph_names_to_index_`.
  /// If we want to append a new subgraph instance name, the function `AddSubgraphName` should be called first.
  /// \param index
  /// \param name
  /// \return
  graphStatus SetSubgraphInstanceName(const uint32_t index, const std::string &name);
  void RemoveSubgraphInstanceName(const std::string &name);

  graphStatus GetSubgraphNameByInstanceName(const std::string &instance_name, std::string &subgraph_name) const;

 protected:
  ProtoAttrMap &MutableAttrMap() override;
  ConstProtoAttrMap &GetAttrMap() const override;

 private:
  bool OpDescMembersAreEqual(const OpDesc &r_op_desc) const;
  bool OpDescAttrsAreEqual(const OpDesc &r_op_desc) const;
  bool OpDescGenTensorDescsAreEqual(const OpDesc &r_op_desc) const;

  OpDescImplPtr impl_;
  friend class OpDescUtils;
  friend class ModelSerializeImp;
  friend class AttrUtils;
  friend class GeAttrValueImp;
  friend class OnnxUtils;
  friend class GraphUtils;
  friend class NodeUtils;
};

using OpDescPtr = OpDesc::OpDescPtr;
using ConstOpDescPtr = OpDesc::ConstOpDescPtr;
using ConstOpDesc = const OpDesc;

class OpDescBuilder {
 public:
  OpDescBuilder(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)) {}
  OpDescBuilder(const OpDescBuilder &) = delete;
  OpDescBuilder &operator=(const OpDescBuilder &) = delete;
  OpDescBuilder(const OpDescBuilder &&) = delete;
  OpDescBuilder &operator=(const OpDescBuilder &&) = delete;
  ~OpDescBuilder() = default;

  /**
   * @brief Add input
   * @param [in] name
   * @return OpDescBuilder
   */
  OpDescBuilder &AddInput(const std::string &name);

  /**
   * @brief Add input
   * @param [in] name
   * @param [in] tensor
   * @return OpDescBuilder
   */
  OpDescBuilder &AddInput(const std::string &name, const GeTensorDesc &tensor);

  /**
   * @brief Add dynamic input
   * @param [in] name
   * @param [in] num
   * @return OpDescBuilder
   */
  OpDescBuilder &AddDynamicInput(const std::string &name, const uint32_t num);

  /**
   * @brief Add dynamic input
   * @param [in] name
   * @param [in] num
   * @param [in] tensor
   * @return OpDescBuilder
   */
  OpDescBuilder &AddDynamicInput(const std::string &name, const uint32_t num, const GeTensorDesc &tensor);

  /**
   * @brief Add output
   * @param [in] name
   * @return OpDescBuilder
   */
  OpDescBuilder &AddOutput(const std::string &name);

  /**
   * @brief Add output
   * @param [in] name
   * @param [in] tensor
   * @return OpDescBuilder
   */
  OpDescBuilder &AddOutput(const std::string &name, const GeTensorDesc &tensor);

  /**
   * @brief Add dynamic output
   * @param [in] name
   * @param [in] num
   * @return OpDescBuilder
   */
  OpDescBuilder &AddDynamicOutput(const std::string &name, const uint32_t num);

  /**
   * @brief Add dynamic output
   * @param [in] name
   * @param [in] num
   * @param [in] tensor
   * @return OpDescBuilder
   */
  OpDescBuilder &AddDynamicOutput(const std::string &name, const uint32_t num, const GeTensorDesc &tensor);

  /**
   * @brief Build op_desc
   * @return OpDescPtr
   */
  OpDescPtr Build();

 private:
  std::string name_;
  std::string type_;
  std::vector<std::pair<std::string, GeTensorDesc>> inputs_;
  std::vector<std::pair<std::string, GeTensorDesc>> outputs_;
};
}  // namespace ge
#endif  // INC_GRAPH_OP_DESC_H_
