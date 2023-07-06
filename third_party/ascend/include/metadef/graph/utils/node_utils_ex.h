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

#ifndef __INC_METADEF_NODE_UTILS_EX_H
#define __INC_METADEF_NODE_UTILS_EX_H

#include "graph/node.h"
#include "graph/op_desc.h"

namespace ge {
class NodeUtilsEx {
 public:
  // Detach from Node
  static graphStatus Verify(const NodePtr &node);
  static graphStatus InferShapeAndType(const NodePtr &node);
  static graphStatus InferOriginFormat(const NodePtr &node);
  // Detach from NodeUtils
  static ConstNodePtr GetNodeFromOperator(const Operator &op);
 private:
  static graphStatus IsInputsValid(const NodePtr &node);
};
} // namespace ge
#endif // __INC_METADEF_NODE_UTILS_EX_H
