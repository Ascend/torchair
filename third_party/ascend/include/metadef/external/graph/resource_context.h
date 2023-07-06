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

#ifndef INC_EXTERNAL_GRAPH_RESOURCE_CONTEXT_H_
#define INC_EXTERNAL_GRAPH_RESOURCE_CONTEXT_H_

namespace ge {
// For resource op infershape, indicate content stored in resources, shape/dtype etc.
// Op can inherit from this struct and extend more content
struct ResourceContext {
  virtual ~ResourceContext() {}
}; // struct ResourceContext
}  // namespace ge
#endif  // INC_EXTERNAL_GRAPH_RESOURCE_CONTEXT_H_
