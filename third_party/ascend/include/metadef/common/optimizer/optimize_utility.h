/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef INC_COMMON_OPTIMIZER_OPTIMIZE_UTILITY_H_
#define INC_COMMON_OPTIMIZER_OPTIMIZE_UTILITY_H_

#include "common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"

namespace ge {
class OptimizeUtility {
 public:
  virtual ~OptimizeUtility() {}

  // Deprecated: will delete later. Graph infershape util
  virtual Status InferShape(ComputeGraph &compute_graph) = 0;

  // Graph infershape util
  virtual Status InferShape(const ComputeGraphPtr &compute_graph) = 0;

  // Mlti Dims and pre/post process
  virtual Status MultiDimsProcess(const ComputeGraphPtr &compute_graph) {
    (void)compute_graph;
    return SUCCESS;
  }
};
}  // namespace ge
#endif  // INC_COMMON_OPTIMIZER_OPTIMIZE_UTILITY_H_
