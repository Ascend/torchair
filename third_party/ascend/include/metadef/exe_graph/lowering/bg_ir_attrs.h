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

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_IR_ATTRS_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_IR_ATTRS_H_
#include "graph/node.h"
#include "value_holder.h"
namespace gert {
namespace bg {
std::unique_ptr<uint8_t[]> CreateAttrBuffer(const ge::NodePtr &node, size_t &size);
std::unique_ptr<uint8_t[]> CreateAttrBuffer(const ge::NodePtr &node,
                                            const std::vector<ge::AnyValue> &runtime_attrs_list,
                                            size_t &size);
}
}
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_IR_ATTRS_H_
