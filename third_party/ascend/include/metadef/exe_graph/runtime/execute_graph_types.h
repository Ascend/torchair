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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXECUTE_GRAPH_TYPES_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXECUTE_GRAPH_TYPES_H_
#include <cstdlib>
namespace gert {
/**
 * 执行图类型，在一个Model中，包含多张执行图，本枚举定义了所有执行图的类型
 */
enum class ExecuteGraphType {
  kInit,    //!< 初始化图，本张图在图加载阶段执行
  kMain,    //!< 主图，每次执行图时，均执行本张图
  kDeInit,  //!< 去初始化图，在图卸载时，执行本张图
  kNum
};

/**
 * 获取执行图的字符串描述
 * @param type 执行图类型枚举
 * @return
 */
inline const char *GetExecuteGraphTypeStr(const ExecuteGraphType type) {
  if (type >= ExecuteGraphType::kNum) {
    return nullptr;
  }
  constexpr const char *kStrs[static_cast<size_t>(ExecuteGraphType::kNum)] = {"Init", "Main", "DeInit"};
  return kStrs[static_cast<size_t>(type)];
}
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXECUTE_GRAPH_TYPES_H_
