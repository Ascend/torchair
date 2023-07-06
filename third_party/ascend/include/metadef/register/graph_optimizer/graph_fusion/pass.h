/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

/** @defgroup FUSION_PASS_GROUP Fusion Pass Interface */

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_PASS_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_PASS_H_

#include "graph/compute_graph.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {

/** fusion pass
 * @ingroup GRAPH_PASS_GROUP
 * network level pass
 */
template <typename T>
class Pass {
 public:
  virtual ~Pass() {}

  /** execute pass
   *
   * @param [in] graph, the graph waiting for pass level optimization
   * @return SUCCESS, successfully optimized the graph by the pass
   * @return NOT_CHANGED, the graph did not change
   * @return FAILED, fail to modify graph
   */
  virtual Status Run(ge::ComputeGraph &graph) = 0;

  void SetName(const std::string &name) { name_ = name; }

  std::string GetName() { return name_; }

 private:
  std::string name_;
};

}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_PASS_H_
