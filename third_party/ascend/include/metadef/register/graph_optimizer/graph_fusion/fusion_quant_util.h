/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#ifndef INC_FUSION_QUANT_UTIL_H_
#define INC_FUSION_QUANT_UTIL_H_
#include "graph/node.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include <vector>

struct BiasOptimizeEdges {
  ge::InDataAnchorPtr quant_scale;
  ge::InDataAnchorPtr quant_offset;
  ge::InDataAnchorPtr cube_weight;
  ge::InDataAnchorPtr cube_bias;
  ge::InDataAnchorPtr deq_scale;
  bool isValid() {
    return !(cube_weight == nullptr || cube_bias == nullptr || deq_scale == nullptr);
  }
};

namespace fe {
class QuantUtil {
 public:
  static Status BiasOptimizeByEdge(BiasOptimizeEdges &param, std::vector<ge::NodePtr> &fusion_nodes);
  static Status BiasOptimizeByEdge(ge::NodePtr &quant_node, BiasOptimizeEdges &param,
                                   std::vector<ge::NodePtr> &fusion_nodes);
  static Status InsertFixpipeDequantScaleConvert(ge::InDataAnchorPtr deq_scale, std::vector<ge::NodePtr> &fusion_nodes);
  static Status InsertFixpipeDequantScaleConvert(ge::InDataAnchorPtr &deq_scale, ge::InDataAnchorPtr &quant_offset,
                                                 std::vector<ge::NodePtr> &fusion_nodes);
  static Status InsertQuantScaleConvert(ge::InDataAnchorPtr &quant_scale, ge::InDataAnchorPtr &quant_offset,
                                        std::vector<ge::NodePtr> &fusion_nodes);
  static Status InsertRequantScaleConvert(ge::InDataAnchorPtr &req_scale, ge::InDataAnchorPtr &quant_offset,
                                          ge::InDataAnchorPtr &cuba_bias, std::vector<ge::NodePtr> &fusion_nodes);
};
}  // namespace fe
#endif