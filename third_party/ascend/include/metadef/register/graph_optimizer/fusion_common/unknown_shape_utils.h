/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_UNKNOWN_SHAPE_UTILS_H
#define INC_REGISTER_GRAPH_OPTIMIZER_UNKNOWN_SHAPE_UTILS_H
#include "graph/utils/graph_utils.h"
namespace fe {
class UnknownShapeUtils {
public:
  /*
   *  @ingroup fe
   *  @brief   check whether the node is unknown shape.
   *  @param   [in]  input or output tensor.
   *  @return  true: unknown; false: known
   */
  static bool IsUnknownShapeOp(const ge::OpDesc &op_desc);

  /*
   *  @ingroup fe
   *  @brief   check whether the input or output shape contains -2.
   *  @param   op_desc input or output desc.
   *  @return  true: contains; false: not contains
   */
  static bool IsContainUnknownDimNum(const ge::OpDesc &op_desc);

  /*
   *  @brief   check whether the value is -1 or -2
   *  @param   input or ourput shape dim
   *  @return  true: contains; false: not contains
   */
  static bool IsUnknownShapeValue(const int64_t &value);
private:
  static bool IsUnKnownShapeTensor(const ge::OpDesc &op_desc);
};
}  // namespace fe

#endif