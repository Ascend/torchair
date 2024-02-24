/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef INC_GRAPH_MODEL_SERIALIZE_H_
#define INC_GRAPH_MODEL_SERIALIZE_H_

#include <map>
#include <string>
#include "graph/buffer.h"
#include "graph/compute_graph.h"
#include "graph/model.h"
#include "external/ge_common/ge_api_types.h"

namespace ge {
class ModelSerialize {
 public:
  Buffer SerializeModel(const Model &model, const bool not_dump_all = false) const;
  Buffer SerializeSeparateModel(const Model &model, const std::string &path, const bool not_dump_all = false) const;
  Buffer SerializeModel(const Model &model, const std::string &path,
                        const bool is_need_separate, const bool not_dump_all = false) const;
  Status SerializeModel(const Model &model, const bool not_dump_all, proto::ModelDef &model_def) const;

  bool UnserializeModel(const uint8_t *const data, const size_t len, Model &model) const;
  bool UnserializeModel(ge::proto::ModelDef &model_def, Model &model, const std::string &path) const;
  bool UnserializeModel(ge::proto::ModelDef &model_def, Model &model) const;

 private:
  friend class ModelSerializeImp;
  friend class GraphDebugImp;
};
}  // namespace ge
#endif  // INC_GRAPH_MODEL_SERIALIZE_H_
