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
#ifndef INC_50EA5B1AAF3341A28036E698708ADB64_H
#define INC_50EA5B1AAF3341A28036E698708ADB64_H
#include <cstdint>
#include <string>
#include <unordered_set>
#include "graph/ge_error_codes.h"
#include "kernel_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/ascend_string.h"

namespace ge {
class AnyValue;
}  // namespace ge

namespace gert {
struct OpImplKernelRegistry {
  using InferShapeKernelFunc = UINT32 (*)(InferShapeContext *);
  using InferShapeRangeKernelFunc = UINT32 (*)(InferShapeRangeContext *);
  using TilingKernelFunc = UINT32 (*)(TilingContext *);
  using InferDataTypeKernelFunc = UINT32 (*)(InferDataTypeContext *);
  using OpType = ge::AscendString;
  using PrivateAttrList = std::vector<std::pair<ge::AscendString, ge::AnyValue>>;
  using PrivateAttrSet = std::unordered_set<ge::AscendString>;
  using CompileInfoCreatorFunc = void *(*)();
  using CompileInfoDeleterFunc = void (*)(void *);

  struct OpImplFunctions {
    bool HasDataDependency() const {
      return (inputs_dependency != 0U);
    }
    /*
     * param index: must be ir index
     */
    bool IsInputDataDependency(const size_t index) const {
      if (index >= sizeof(inputs_dependency) * kInt64ByteCount) {
        return false;
      }
      return static_cast<bool>(inputs_dependency & static_cast<uint64_t>(1) << index);
    }
    ge::graphStatus SetInputDataDependency(const size_t index) {
      if (index >= sizeof(inputs_dependency) * kInt64ByteCount) {
        return ge::GRAPH_FAILED;
      }
      inputs_dependency |= 1UL << index;
      return ge::GRAPH_SUCCESS;
    }

    InferShapeKernelFunc infer_shape;
    InferShapeRangeKernelFunc infer_shape_range;
    InferDataTypeKernelFunc infer_datatype;
    TilingKernelFunc tiling;
    KernelRegistry::KernelFunc tiling_parse;
    CompileInfoCreatorFunc compile_info_creator;
    CompileInfoDeleterFunc compile_info_deleter;
    size_t max_tiling_data_size = 0;
    uint64_t inputs_dependency = 0;
    static constexpr size_t kInt64ByteCount = 8;
    PrivateAttrList private_attrs;
    // todo 去重和registry没关系，下一步从这里删除，移动到register中实现
    PrivateAttrSet unique_private_attrs;
    uint8_t reserved_0_[7] = {0U};   // Reserved field, 8-byte aligned for unique_private_attrs
    uint8_t reserved_1_[40] = {0U};  // Reserved field, 32+8, do not directly use when only 8-byte left
  };
  virtual ~OpImplKernelRegistry() = default;
  virtual const OpImplFunctions *GetOpImpl(const ge::char_t *op_type) const = 0;
  virtual const PrivateAttrList &GetPrivateAttrs(const ge::char_t *op_type) const = 0;
};
}  // namespace gert

#endif
