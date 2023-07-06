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

#ifndef METADEF_CXX_INC_EXE_GRAPH_LOWERING_GENERATE_EXE_GRAPH_H_
#define METADEF_CXX_INC_EXE_GRAPH_LOWERING_GENERATE_EXE_GRAPH_H_
#include <vector>
#include "value_holder.h"
#include "lowering_global_data.h"
#include "graph/compute_graph.h"
namespace gert {
namespace bg {
class GenerateExeGraph {
 public:
  struct ExeGraphGenerator {
    using InferShapeFunc = std::vector<ValueHolderPtr> (*)(const ge::NodePtr &node,
                                                           const std::vector<ValueHolderPtr> &shapes);
    using AllocOutputMemoryFunc = std::vector<ValueHolderPtr> (*)(TensorPlacement placement, const ge::NodePtr &node,
                                                                  const std::vector<ValueHolderPtr> &output_sizes,
                                                                  LoweringGlobalData &global_data);
    using CalcTensorSizeFunc = std::vector<ValueHolderPtr> (*)(const ge::NodePtr &node,
                                                               const std::vector<ValueHolderPtr> &output_shapes);

    InferShapeFunc infer_shape;
    AllocOutputMemoryFunc alloc_output_memory;
    CalcTensorSizeFunc calc_tensor_size;
  };

 public:
  static std::vector<ValueHolderPtr> InferShape(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &shapes) {
    if (generator_.infer_shape == nullptr) {
      return {};
    }
    return generator_.infer_shape(node, shapes);
  }
  static std::vector<ValueHolderPtr> AllocOutputMemory(TensorPlacement placement, const ge::NodePtr &node,
                                                       const std::vector<ValueHolderPtr> &output_sizes,
                                                       LoweringGlobalData &global_data) {
    if (generator_.alloc_output_memory == nullptr) {
      return {};
    }
    return generator_.alloc_output_memory(placement, node, output_sizes, global_data);
  }
  static std::vector<ValueHolderPtr> CalcTensorSize(const ge::NodePtr &node,
                                                    const std::vector<ValueHolderPtr> &output_shapes) {
    if (generator_.calc_tensor_size == nullptr) {
      return {};
    }
    return generator_.calc_tensor_size(node, output_shapes);
  }

  static void AddBuilderImplement(ExeGraphGenerator generator) {
    generator_ = generator;
  }

 private:
  static ExeGraphGenerator generator_;
};
}  // namespace bg
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_LOWERING_GENERATE_EXE_GRAPH_H_
