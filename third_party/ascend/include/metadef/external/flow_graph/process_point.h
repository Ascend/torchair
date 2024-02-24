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

#ifndef INC_EXTERNAL_FLOW_GRAPH_PROCESS_POINT_H_
#define INC_EXTERNAL_FLOW_GRAPH_PROCESS_POINT_H_

#include <cstdint>
#include <functional>
#include <map>
#include "graph/graph.h"
#include "graph/types.h"

namespace ge {
namespace dflow {
using GraphBuilder = std::function<ge::Graph()>;
enum class ProcessPointType {
  FUNCTION = 0,
  GRAPH = 1,
  INNER = 2,
  INVALID = 3,
};

class ProcessPointImpl;
class ProcessPoint {
public:
  virtual ~ProcessPoint();
  ProcessPointType GetProcessPointType() const;
  const char_t *GetProcessPointName() const;
  const char_t *GetCompileConfig() const;
  virtual void Serialize(ge::AscendString &str) const = 0;

protected:
  ProcessPoint(const char_t *pp_name, ProcessPointType pp_type);
  void SetCompileConfigFile(const char_t *json_file_path);

private:
  std::shared_ptr<ProcessPointImpl> impl_;
};

class GraphPpImpl;
class GraphPp : public ProcessPoint {
public:
  GraphPp(const char_t *pp_name, const GraphBuilder &builder);
  ~GraphPp() override;
  GraphPp &SetCompileConfig(const char_t *json_file_path);
  GraphBuilder GetGraphBuilder() const;
  void Serialize(ge::AscendString &str) const override;
private:
  std::shared_ptr<GraphPpImpl> impl_;
};

class FunctionPpImpl;
class FunctionPp : public ProcessPoint {
public:
  explicit FunctionPp(const char_t *pp_name);
  ~FunctionPp() override;
  FunctionPp &SetCompileConfig(const char_t *json_file_path);
  FunctionPp &SetInitParam(const char_t *attr_name, const ge::AscendString &value);
  FunctionPp &SetInitParam(const char_t *attr_name, const char_t *value);
  FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<ge::AscendString> &value);
  FunctionPp &SetInitParam(const char_t *attr_name, const int64_t &value);
  FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<int64_t> &value);
  FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<std::vector<int64_t>> &value);
  FunctionPp &SetInitParam(const char_t *attr_name, const float &value);
  FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<float> &value);
  FunctionPp &SetInitParam(const char_t *attr_name, const bool &value);
  FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<bool> &value);
  FunctionPp &SetInitParam(const char_t *attr_name, const ge::DataType &value);
  FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<ge::DataType> &value);
  FunctionPp &AddInvokedClosure(const char_t *name, const GraphPp &graph_pp);
  FunctionPp &AddInvokedClosure(const char_t *name, const ProcessPoint &pp);
  const std::map<const std::string, const GraphPp> &GetInvokedClosures() const;
  void Serialize(ge::AscendString &str) const override;
private:
  std::shared_ptr<FunctionPpImpl> impl_;
};
} // namespace dflow
} // namespace ge
#endif // INC_EXTERNAL_FLOW_GRAPH_PROCESS_POINT_H_