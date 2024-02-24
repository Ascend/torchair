/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef OP_DEF_H
#define OP_DEF_H

#include <iostream>
#include <vector>
#include <memory>
#include "register/op_impl_registry.h"
#include "graph/operator_reg.h"

namespace optiling {
#define FUNC_CHECK_SUPPORTED "check_supported"
#define FUNC_OP_SELECT_FORMAT "op_select_format"
#define FUNC_GET_OP_SUPPORT_INFO "get_op_support_info"
#define FUNC_GET_SPECIFIC_INFO "get_op_specific_info"

using OP_CHECK_FUNC = ge::graphStatus (*)(const ge::Operator &op, ge::AscendString &result);

using PARAM_GENERALIZE_FUNC = ge::graphStatus (*)(const ge::Operator &op, const ge::AscendString &generalize_config,
                                      ge::AscendString &generalized_op_params);

class OpCheckFuncHelper {
public:
  OpCheckFuncHelper(const ge::AscendString &check_type, const ge::AscendString &op_type, OP_CHECK_FUNC func);

  OpCheckFuncHelper(const ge::AscendString &op_type, PARAM_GENERALIZE_FUNC func);
};
}

namespace ops {
enum Option { IGNORE = 0, OPTIONAL = 1, REQUIRED = 2, DYNAMIC = 3, VIRTUAL = 4 };

enum class AttrDataType {
  ATTR_DT_BOOL = 0,
  ATTR_DT_FLOAT = 1,
  ATTR_DT_INT = 2,
  ATTR_DT_STR = 3,
  ATTR_DT_LIST_BOOL = 4,
  ATTR_DT_LIST_FLOAT = 5,
  ATTR_DT_LIST_INT = 6,
  ATTR_DT_LIST_LIST_INT = 7,
  ATTR_DT_MAX
};

enum class ItemFindStatus { ITEM_FIND = 0, ITEM_NOEXIST = 1 };

class OpParamDefImpl;
class OpParamDef {
public:
  explicit OpParamDef(const char *name);
  OpParamDef(const OpParamDef &def);
  ~OpParamDef();
  OpParamDef &operator=(const OpParamDef &def);
  bool operator==(const OpParamDef &def) const;
  void MergeParam(const OpParamDef &def);
  OpParamDef &ParamType(Option param_type);
  OpParamDef &DataType(std::vector<ge::DataType> types);
  OpParamDef &Format(std::vector<ge::Format> formats);
  OpParamDef &UnknownShapeFormat(std::vector<ge::Format> formats);
  OpParamDef &ValueDepend(Option value_depend);
  OpParamDef &IgnoreContiguous(void);
  OpParamDef &AutoContiguous();
  OpParamDef &Scalar();
  OpParamDef &ScalarList();
  OpParamDef &To(const ge::DataType type);
  OpParamDef &To(const char *name);
  OpParamDef &Version(uint32_t version);
  ge::AscendString &GetParamName(void);
  Option GetParamType(void);
  std::vector<ge::DataType> &GetDataTypes(void);
  std::vector<ge::Format> &GetFormats(void);
  std::vector<ge::Format> &GetUnknownShapeFormats(void);
  ge::AscendString &GetValueDepend(void);
  bool GetIgnoreContiguous(void);
  bool GetAutoContiguous(void);
  bool IsScalar(void);
  bool IsScalarList(void);
  ge::AscendString &GetScalarName(void);
  ge::DataType GetScalarType(void);
  uint32_t GetVersion(void);

private:
  std::unique_ptr<OpParamDefImpl> impl_;
};

class OpAttrDefImpl;
class OpAttrDef {
public:
  explicit OpAttrDef(const char *name);
  OpAttrDef(const OpAttrDef &attr_def);
  ~OpAttrDef();
  OpAttrDef &operator=(const OpAttrDef &attr_def);
  bool operator==(const OpAttrDef &attr_def) const;
  OpAttrDef &AttrType(Option attr_type);
  OpAttrDef &Bool(void);
  OpAttrDef &Bool(bool value);
  OpAttrDef &Float(void);
  OpAttrDef &Float(float value);
  OpAttrDef &Int(void);
  OpAttrDef &Int(int64_t value);
  OpAttrDef &String(void);
  OpAttrDef &String(const char *value);
  OpAttrDef &ListBool(void);
  OpAttrDef &ListBool(std::vector<bool> value);
  OpAttrDef &ListFloat(void);
  OpAttrDef &ListFloat(std::vector<float> value);
  OpAttrDef &ListInt(void);
  OpAttrDef &ListInt(std::vector<int64_t> value);
  OpAttrDef &ListListInt(void);
  OpAttrDef &ListListInt(std::vector<std::vector<int64_t>> value);
  OpAttrDef &Version(uint32_t version);
  ge::AscendString &GetName(void) const;
  bool IsRequired(void);
  ge::AscendString &GetCfgDataType(void) const;
  ge::AscendString &GetProtoDataType(void) const;
  ge::AscendString &GetAttrDefaultVal(const char *brac);
  uint32_t GetVersion(void);

private:
  std::unique_ptr<OpAttrDefImpl> impl_;
};

class OpAICoreConfigImpl;
class OpAICoreConfig {
public:
  OpAICoreConfig();
  OpAICoreConfig(const OpAICoreConfig &aicore_config);
  ~OpAICoreConfig();
  OpAICoreConfig &operator=(const OpAICoreConfig &aicore_config);
  OpParamDef &Input(const char *name);
  OpParamDef &Output(const char *name);
  OpAICoreConfig &DynamicCompileStaticFlag(bool flag);
  OpAICoreConfig &DynamicFormatFlag(bool flag);
  OpAICoreConfig &DynamicRankSupportFlag(bool flag);
  OpAICoreConfig &DynamicShapeSupportFlag(bool flag);
  OpAICoreConfig &NeedCheckSupportFlag(bool flag);
  OpAICoreConfig &PrecisionReduceFlag(bool flag);
  OpAICoreConfig &ExtendCfgInfo(const char *key, const char *value);
  std::vector<OpParamDef> &GetInputs(void);
  std::vector<OpParamDef> &GetOutputs(void);
  std::vector<ge::AscendString> &GetCfgKeys(void);
  std::map<ge::AscendString, ge::AscendString> &GetCfgInfo(void);
  ge::AscendString &GetConfigValue(const char *key);

private:
  void AddCfgItem(const char *key, const char *value);
  std::unique_ptr<OpAICoreConfigImpl> impl_;
};

class OpAICoreDefImpl;
class OpAICoreDef {
public:
  OpAICoreDef();
  OpAICoreDef(const OpAICoreDef &aicore_def);
  ~OpAICoreDef();
  OpAICoreDef &operator=(const OpAICoreDef &aicore_def);
  OpAICoreDef &SetTiling(gert::OpImplKernelRegistry::TilingKernelFunc func);
  OpAICoreDef &SetCheckSupport(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetOpSelectFormat(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetOpSupportInfo(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetOpSpecInfo(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetParamGeneralize(optiling::PARAM_GENERALIZE_FUNC func);
  gert::OpImplKernelRegistry::TilingKernelFunc &GetTiling(void);
  optiling::OP_CHECK_FUNC &GetCheckSupport(void);
  optiling::OP_CHECK_FUNC &GetOpSelectFormat(void);
  optiling::OP_CHECK_FUNC &GetOpSupportInfo(void);
  optiling::OP_CHECK_FUNC &GetOpSpecInfo(void);
  optiling::PARAM_GENERALIZE_FUNC &GetParamGeneralize(void);
  OpAICoreDef &AddConfig(const char *soc);
  OpAICoreDef &AddConfig(const char *soc, OpAICoreConfig &aicore_config);
  std::map<ge::AscendString, OpAICoreConfig> &GetAICoreConfigs(void);

private:
  void Log(const char *op_type, const char *info) const;
  std::unique_ptr<OpAICoreDefImpl> impl_;
};

class OpDefImpl;
class OpDef {
public:
  explicit OpDef(const char *type);
  OpDef(const OpDef &op_def);
  ~OpDef();
  OpDef &operator=(const OpDef &op_def);
  OpParamDef &Input(const char *name);
  OpParamDef &Output(const char *name);
  OpAttrDef &Attr(const char *name);
  OpDef &SetInferShape(gert::OpImplKernelRegistry::InferShapeKernelFunc func);
  OpDef &SetInferShapeRange(gert::OpImplKernelRegistry::InferShapeRangeKernelFunc func);
  OpDef &SetInferDataType(gert::OpImplKernelRegistry::InferDataTypeKernelFunc func);
  gert::OpImplKernelRegistry::InferShapeKernelFunc &GetInferShape(void);
  gert::OpImplKernelRegistry::InferShapeRangeKernelFunc &GetInferShapeRange(void);
  gert::OpImplKernelRegistry::InferDataTypeKernelFunc &GetInferDataType(void);
  void SetWorkspaceFlag(bool flag);
  ge::AscendString &GetOpType(void);
  std::vector<OpParamDef> &GetInputs(void);
  std::vector<OpParamDef> &GetOutputs(void);
  std::vector<OpAttrDef> &GetAttrs(void);
  std::vector<OpParamDef> GetMergeInputs(OpAICoreConfig &aicore_config);
  std::vector<OpParamDef> GetMergeOutputs(OpAICoreConfig &aicore_config);
  bool GetWorkspaceFlag(void);
  OpAICoreDef &AICore(void);

private:
  void MergeParam(std::vector<OpParamDef> &merge, std::vector<OpParamDef> &aicore_params) const;
  void CheckParam(std::vector<OpParamDef> &params) const;
  ItemFindStatus FindAttr(const char *name, OpAttrDef **attr);
  OpAttrDef &AddAttr(OpAttrDef &attr);
  OpAttrDef &GetOrCreateAttr(const char *name);
  std::unique_ptr<OpDefImpl> impl_;
};
}  // namespace ops

#endif
