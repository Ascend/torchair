/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef INC_GRAPH_UTILS_ANCHOR_UTILS_H_
#define INC_GRAPH_UTILS_ANCHOR_UTILS_H_

#include "graph/anchor.h"
#include "graph/node.h"

namespace ge {
class AnchorUtils {
 public:
  // Get anchor status
  static AnchorStatus GetStatus(const DataAnchorPtr &data_anchor);

  // Set anchor status
  static graphStatus SetStatus(const DataAnchorPtr &data_anchor, const AnchorStatus anchor_status);

  static int32_t GetIdx(const AnchorPtr &anchor);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_ANCHOR_UTILS_H_
