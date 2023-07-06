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
#include <utility>
#include <memory>
#include "register/op_impl_registry.h"
#include "register/op_check.h"
#include "graph/operator_reg.h"

namespace ops {
enum Option { IGNORE = 0, OPTIONAL = 1, REQUIRED = 2, DYNAMIC = 3 };

enum AttrDataType {
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

enum ItemFindStatus { ITEM_FIND = 0, ITEM_NOEXIST = 1 };

class OpParamDefImpl;
class OpParamDef {
public:
  OpParamDef(const char *name);
  OpParamDef(const OpParamDef &def);
  ~OpParamDef();
  OpParamDef &operator=(const OpParamDef &def);
  bool operator==(const OpParamDef &def) const;
  void MergeParam(const OpParamDef &def);
  OpParamDef &ParamType(Option param_type);
  OpParamDef &DataType(std::vector<ge::DataType> types);
  OpParamDef &Format(std::vector<ge::Format> formats);
  OpParamDef &UnknownShapeFormat(std::vector<ge::Format> formats);
  OpParamDef &NeedCompile(bool need_compile);
  OpParamDef &ReshapeType(const char *reshape_type);
  OpParamDef &ValueDepend(Option value_depend);
  ge::AscendString &GetParamName(void);
  Option GetParamType(void);
  std::vector<ge::DataType> &GetDataTypes(void);
  std::vector<ge::Format> &GetFormats(void);
  std::vector<ge::Format> &GetUnknownShapeFormats(void);
  ge::AscendString &GetNeedCompile(void);
  ge::AscendString &GetReshapeType(void);
  ge::AscendString &GetValueDepend(void);

private:
  std::unique_ptr<OpParamDefImpl> impl_;
};

class OpAttrDefImpl;
class OpAttrDef {
public:
  OpAttrDef(const char *name);
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
  ge::AscendString &GetName(void) const;
  bool IsRequired(void);
  ge::AscendString &GetCfgDataType(void) const;
  ge::AscendString &GetProtoDataType(void) const;
  ge::AscendString &GetAttrDefaultVal(const char *brac);

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
  OpAICoreConfig &AsyncFlag(bool flag);
  OpAICoreConfig &DynamicCompileStaticFlag(bool flag);
  OpAICoreConfig &DynamicFormatFlag(bool flag);
  OpAICoreConfig &DynamicRankSupportFlag(bool flag);
  OpAICoreConfig &DynamicShapeSupportFlag(bool flag);
  OpAICoreConfig &HeavyOpFlag(bool flag);
  OpAICoreConfig &NeedCheckSupportFlag(bool flag);
  OpAICoreConfig &OpPattern(const char *pattern);
  OpAICoreConfig &PrecisionReduceFlag(bool flag);
  OpAICoreConfig &RangeLimitValue(const char *value);
  OpAICoreConfig &SlicePatternValue(const char *value);
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
  OpAICoreDef &SetTilingParse(gert::OpImplRegister::TilingParseFunc func);
  OpAICoreDef &SetCompileInfoCreator(gert::OpImplKernelRegistry::CompileInfoCreatorFunc func);
  OpAICoreDef &SetCompileInfoDeleter(gert::OpImplKernelRegistry::CompileInfoDeleterFunc func);
  OpAICoreDef &SetCheckSupport(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetOpSelectFormat(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetOpSupportInfo(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetOpSpecInfo(optiling::OP_CHECK_FUNC func);
  OpAICoreDef &SetParamGeneralize(optiling::PARAM_GENERALIZE_FUNC func);
  gert::OpImplKernelRegistry::TilingKernelFunc &GetTiling(void);
  gert::OpImplRegister::TilingParseFunc &GetTilingParse(void);
  gert::OpImplKernelRegistry::CompileInfoCreatorFunc &GetCompileInfoCreator(void);
  gert::OpImplKernelRegistry::CompileInfoDeleterFunc &GetCompileInfoDeleter(void);
  optiling::OP_CHECK_FUNC &GetCheckSupport(void);
  optiling::OP_CHECK_FUNC &GetOpSelectFormat(void);
  optiling::OP_CHECK_FUNC &GetOpSupportInfo(void);
  optiling::OP_CHECK_FUNC &GetOpSpecInfo(void);
  optiling::PARAM_GENERALIZE_FUNC &GetParamGeneralize(void);
  void AddConfig(const char *soc, OpAICoreConfig &aicore_config);
  std::map<ge::AscendString, OpAICoreConfig> &GetAICoreConfigs(void);
  template<class T>
  void OpTilingPost(const char *op_type) {
    this->Log(op_type, "do optiling post");
    gert::OpImplRegisterV2 impl(op_type);
    impl.Tiling(this->GetTiling());
    impl.TilingParse<T>(this->GetTilingParse());
    gert::OpImplRegisterV2 implReg(impl);
  }
  void OpCheckPost(const char *op_type);

private:
  void Log(const char *op_type, const char *info);
  std::unique_ptr<OpAICoreDefImpl> impl_;
};

class OpDefImpl;
class OpDef {
public:
  OpDef(const char *type);
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
  void OpProtoPost(const char *op_type);
  OpAICoreDef &AICore(void);

private:
  void MergeParam(std::vector<OpParamDef> &merge, std::vector<OpParamDef> &aicore_params);
  void CheckParam(std::vector<OpParamDef> &params);
  int FindAttr(const char *name, OpAttrDef **attr);
  OpAttrDef &AddAttr(OpAttrDef &attr);
  OpAttrDef &GetOrCreateAttr(const char *name);
  std::unique_ptr<OpDefImpl> impl_;
};
}  // namespace ops

#endif
