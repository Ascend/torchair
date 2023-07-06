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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_RANGE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_RANGE_CONTEXT_H_
#include <type_traits>
#include "range.h"
#include "tensor.h"
#include "runtime_attrs.h"
#include "extended_kernel_context.h"
namespace gert {
using TensorRange = Range<Tensor>;
/**
 * InferShapeRange kernel的context
 */
class InferShapeRangeContext : public ExtendedKernelContext {
 public:
  /**
   * 根据输入index，获取输入shape range指针
   * @param index 输入index
   * @return 输入shape range指针，index非法时，返回空指针
   */
  const Range<Shape> *GetInputShapeRange(const size_t index) const {
    return GetInputPointer<Range<Shape>>(index);
  }
  /**
   * 根据输入index，获取输出tensor指针
   *
   * **注意：只有在`IMPL_OP`实现算子时， 将对应输入设置为数据依赖后，才可以调用此接口获取tensor，否则行为是未定义的。**
   * @param index 输入index
   * @return 输入tensor指针，index非法时，返回空指针
   */
  const TensorRange *GetInputTensorRange(const size_t index) const {
    return GetInputPointer<TensorRange>(index);
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor range指针
   * @param ir_index IR原型定义中的index
   * @return tensor range指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const TensorRange *GetOptionalInputTensorRange(const size_t ir_index) const {
    return GetDynamicInputPointer<TensorRange>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入tensor指针
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return tensor range指针，index或relative_index非法时，返回空指针
   */
  const TensorRange *GetDynamicInputTensorRange(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<TensorRange>(ir_index, relative_index);
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入shape range指针
   * @param ir_index IR原型定义中的index
   * @return shape range指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Range<Shape> *GetOptionalInputShapeRange(const size_t ir_index) const {
    return GetDynamicInputPointer<Range<Shape>>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入shape指针
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return shape range指针，index或relative_index非法时，返回空指针
   */
  const Range<Shape> *GetDynamicInputShapeRange(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Range<Shape>>(ir_index, relative_index);
  }

  /**
   * 根据输出index，获取输出shape range指针
   * @param index 输出index
   * @return 输出shape range指针，index非法时，返回空指针
   */
  Range<Shape> *GetOutputShapeRange(const size_t index) {
    return GetOutputPointer<Range<Shape>>(index);
  }
};
static_assert(std::is_standard_layout<InferShapeRangeContext>::value, "The class InferShapeContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_RANGE_CONTEXT_H_
