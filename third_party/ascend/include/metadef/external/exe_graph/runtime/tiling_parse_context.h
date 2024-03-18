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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_TILING_PARSE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_TILING_PARSE_CONTEXT_H_
#include "exe_graph/runtime/extended_kernel_context.h"
#include "graph/types.h"

namespace fe {
class PlatFormInfos;
}  // namespace fe

namespace gert {
class TilingParseContext : public ExtendedKernelContext {
 public:
  /**
   * 获取算子编译产生的json字符串
   * @return json字符串
   */
  const ge::char_t *GetCompiledJson() const {
    return GetInputValue<const char *>(0);
  }
  /**
   * 获取`CompiledInfo`实例
   * @tparam T 实例类型，该类型需要与`IMPL_OP`注册时TilingParse的类型一致
   * @return 指向`CompiledInfo`实例的指针
   */
  template<typename T>
  auto GetCompiledInfo() -> T* {
    auto av = GetOutput(0);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<T *>();
  }
  /**
   * 获取 fe::PlatFormInfos 指针
   * @return fe::PlatFormInfos 指针
   */
  fe::PlatFormInfos *GetPlatformInfo() const {
    return GetInputValue<fe::PlatFormInfos *>(1);
  }
};
static_assert(std::is_standard_layout<TilingParseContext>::value, "The class TilingParseContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_TILING_PARSE_CONTEXT_H_
