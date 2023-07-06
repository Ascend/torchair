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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SHAPE_UTILS_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SHAPE_UTILS_H_
#include <string>
#include <sstream>
#include "exe_graph/runtime/shape.h"
namespace gert {
extern const Shape g_vec_1_shape;
/**
 * 确保返回的shape是非scalar的。
 * 当一个shape的dim num为0时，此shape被认为表达了一个scalar。
 * 本函数在接受一个非scalar的shape时，会返回原有shape；在接收到scalar shape时，会返回返回一个{1}的vector shape
 * @param in_shape 输入shape
 * @return 保证非scalar的shape
 */
inline const Shape &EnsureNotScalar(const Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}
/**
 * 返回shape的字符串，本函数性能较低，不可以在执行时的正常流程中使用
 * @param shape 需要转为字符串的shape实例
 * @param join_char 每个Dim的间隔，默认为`,`
 * @return 转好的字符串
 */
inline std::string ShapeToString(const Shape &shape, const char *join_char = ",") {
  if (join_char == nullptr) {
    join_char = ",";
  }
  std::stringstream ss;
  for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
    if (i > 0U) {
      ss << join_char;
    }
    ss << shape[i];
  }
  return ss.str();
}
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SHAPE_UTILS_H_
