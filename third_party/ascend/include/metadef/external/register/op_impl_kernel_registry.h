/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_50EA5B1AAF3341A28036E698708ADB64_H
#define INC_50EA5B1AAF3341A28036E698708ADB64_H
#include <cstdint>
#include <string>
#include <unordered_set>
#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/base_type.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/ascend_string.h"

namespace ge {
class AnyValue;
}  // namespace ge

namespace gert {
class TilingParseContext;
struct OpImplKernelRegistry {
  using InferShapeKernelFunc = UINT32 (*)(InferShapeContext *);
  using InferShapeRangeKernelFunc = UINT32 (*)(InferShapeRangeContext *);
  using TilingKernelFunc = UINT32 (*)(TilingContext *);
  using InferDataTypeKernelFunc = UINT32 (*)(InferDataTypeContext *);
  using GenSimplifiedKeyKernelFunc = UINT32 (*)(TilingContext *, ge::char_t *);
  // aclnn接口的原型，入参含义：
  // OpExecuteContext：保存算子的Input，Output，Attr信息
  using OpExecuteFunc = UINT32 (*)(OpExecuteContext *);
  using OpType = ge::AscendString;
  using PrivateAttrList = std::vector<std::pair<ge::AscendString, ge::AnyValue>>;
  using PrivateAttrSet = std::unordered_set<ge::AscendString>;
  using CompileInfoCreatorFunc = void *(*) ();
  using CompileInfoDeleterFunc = void (*)(void *);
  using KernelFunc = UINT32 (*)(KernelContext *context);
  using TilingParseFunc = UINT32 (*)(TilingParseContext *context);

  struct OpImplFunctions {
    bool HasDataDependency() const {
      return (inputs_dependency != 0U);
    }
    /*
     * param index: must be ir index
     */
    bool IsInputDataDependency(const size_t index) const {
      if (index >= sizeof(inputs_dependency) * kByteBitCount) {
        return false;
      }
      return static_cast<bool>(inputs_dependency & static_cast<uint64_t>(1) << index);
    }
    ge::graphStatus SetInputDataDependency(const size_t index) {
      if (index >= sizeof(inputs_dependency) * kByteBitCount) {
        return ge::GRAPH_FAILED;
      }
      inputs_dependency |= 1UL << index;
      return ge::GRAPH_SUCCESS;
    }

    bool HasHostInput() const {
      return (host_inputs != 0UL);
    }
    /*
     * param index: must be ir index
     */
    bool IsHostInput(const size_t index) const {
      if (index >= (sizeof(host_inputs) * kByteBitCount)) {
        return false;
      }
      return static_cast<bool>(host_inputs & (static_cast<uint64_t>(1) << index));
    }
    ge::graphStatus SetHostInputs(const size_t index) {
      if (index >= (sizeof(host_inputs) * kByteBitCount)) {
        return ge::GRAPH_FAILED;
      }
      host_inputs |= 1UL << index;
      return ge::GRAPH_SUCCESS;
    }

    bool HasTilingInputDataDependency() const {
      return (tiling_dependency != 0UL);
    }
    /*
     * param index: must be ir index
     */
    bool IsTilingInputDataDependency(const size_t index) const {
      if (index >= (sizeof(tiling_dependency) * kByteBitCount)) {
        return false;
      }
      return static_cast<bool>(tiling_dependency & (static_cast<uint64_t>(1) << index));
    }
    ge::graphStatus SetTilingInputDataDependency(const size_t index) {
      if (index >= (sizeof(tiling_dependency) * kByteBitCount)) {
        return ge::GRAPH_FAILED;
      }
      tiling_dependency |= 1UL << index;
      return ge::GRAPH_SUCCESS;
    }

    bool IsSupportTilingDependencyPlacement(const uint32_t placement) const {
      if (static_cast<size_t>(placement) >= (sizeof(tiling_dependency_placements) * kByteBitCount)) {
        return false;
      }

      return static_cast<bool>(tiling_dependency_placements & (static_cast<uint8_t>(1U) << placement));
    }

    ge::graphStatus SetTilingDependencyPlacement(const uint32_t placement) {
      if (static_cast<size_t>(placement) >= (sizeof(tiling_dependency_placements) * kByteBitCount)) {
        return ge::GRAPH_FAILED;
      }
      tiling_dependency_placements |= (static_cast<uint8_t>(1U) << placement);
      return ge::GRAPH_SUCCESS;
    }

    InferShapeKernelFunc infer_shape;
    InferShapeRangeKernelFunc infer_shape_range;
    InferDataTypeKernelFunc infer_datatype;
    TilingKernelFunc tiling;
    KernelFunc tiling_parse;
    CompileInfoCreatorFunc compile_info_creator;
    CompileInfoDeleterFunc compile_info_deleter;
    size_t max_tiling_data_size = 0UL;
    uint64_t inputs_dependency = 0UL;
    static constexpr size_t kByteBitCount = 8UL;
    PrivateAttrList private_attrs;
    // todo 去重和registry没关系，下一步从这里删除，移动到register中实现
    PrivateAttrSet unique_private_attrs;
    uint64_t host_inputs = 0UL;
    OpExecuteFunc op_execute_func;
    uint64_t tiling_dependency = 0UL;
    GenSimplifiedKeyKernelFunc gen_simplifiedkey;
    uint8_t tiling_dependency_placements = 0U;
    uint8_t reserved_0_[6] = {0U};   // Reserved field, 8-byte aligned for unique_private_attrs
    uint8_t reserved_1_[8] = {0U};  // Reserved field, 8, do not directly use when only 8-byte left
  };
};
}  // namespace gert
#endif
