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

#ifndef GRAPH_IR_DEFINITIONS_RECOVER_H_
#define GRAPH_IR_DEFINITIONS_RECOVER_H_

#include <string>
#include "graph/compute_graph.h"

namespace ge {
ge::graphStatus RecoverIrDefinitions(const ge::ComputeGraphPtr &graph, const vector<std::string> &attr_names = {});
ge::graphStatus RecoverOpDescIrDefinition(const ge::OpDescPtr &desc, const std::string &op_type = "");
} // namespace ge
#endif  // GRAPH_IR_DEFINITIONS_RECOVER_H_
