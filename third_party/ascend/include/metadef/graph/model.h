/*
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

#ifndef INC_GRAPH_MODEL_H_
#define INC_GRAPH_MODEL_H_

#include <memory>
#include <string>
#include "detail/attributes_holder.h"
#include "graph/ge_attr_value.h"
#include "graph/compute_graph.h"

namespace ge {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Model : public AttrHolder {
 public:
  Model();

  ~Model() override = default;

  Model(const std::string &name, const std::string &custom_version);

  std::string GetName() const;
  void SetName(const std::string &name);

  uint32_t GetVersion() const;

  void SetVersion(const uint32_t version) { version_ = version; }

  std::string GetPlatformVersion() const;

  void SetPlatformVersion(const std::string version) { platform_version_ = version; }

  const ComputeGraphPtr GetGraph() const;

  void SetGraph(const ComputeGraphPtr &graph);

  void SetAttr(const ProtoAttrMap &attrs);

  using AttrHolder::GetAllAttrNames;
  using AttrHolder::GetAllAttrs;
  using AttrHolder::GetAttr;
  using AttrHolder::HasAttr;
  using AttrHolder::SetAttr;

  graphStatus Save(Buffer &buffer, const bool is_dump = false) const;
  graphStatus Save(proto::ModelDef &model_def, const bool is_dump = false) const;

  graphStatus SaveToFile(const std::string &file_name) const;
  // Model will be rewrite
  static graphStatus Load(const uint8_t *data, size_t len, Model &model);
  graphStatus Load(ge::proto::ModelDef &model_def);
  graphStatus LoadFromFile(const std::string &file_name);

  bool IsValid() const;

 protected:
  ConstProtoAttrMap &GetAttrMap() const override;
  ProtoAttrMap &MutableAttrMap() override;

 private:
  void Init();
  graphStatus Load(ge::proto::ModelDef &model_def, const std::string &path);
  graphStatus Save(Buffer &buffer, const std::string &path, const bool is_dump = false) const;
  AttrStore attrs_;
  friend class ModelSerializeImp;
  friend class GraphDebugImp;
  friend class OnnxUtils;
  friend class ModelHelper;
  friend class ModelBuilder;
  std::string name_;
  uint32_t version_;
  std::string platform_version_{""};
  ComputeGraphPtr graph_;
};
using ModelPtr = std::shared_ptr<ge::Model>;
}  // namespace ge

#endif  // INC_GRAPH_MODEL_H_
