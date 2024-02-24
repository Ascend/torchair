/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef GE_RUNTIME_TILING_PARSE_CONTEXT_BUILDER_H_
#define GE_RUNTIME_TILING_PARSE_CONTEXT_BUILDER_H_

#include "exe_graph/runtime/kernel_context.h"
#include "graph/operator.h"
#include "exe_graph/runtime/kernel_run_context_builder.h"
#include "register/op_impl_kernel_registry.h"

namespace gert {
class TilingParseContextBuilder {
 public:
  TilingParseContextBuilder &CompileJson(const ge::char_t *compile_json);
  TilingParseContextBuilder &PlatformInfo(void *platform_info);
  TilingParseContextBuilder &CompileInfoCreatorFunc(OpImplKernelRegistry::CompileInfoCreatorFunc create_func);
  TilingParseContextBuilder &CompileInfoDeleterFunc(OpImplKernelRegistry::CompileInfoDeleterFunc delete_func);
  KernelContextHolder Build(const ge::Operator &op);

 private:
  void *compile_json_{ nullptr };
  void *platform_info_{ nullptr };
  OpImplKernelRegistry::CompileInfoCreatorFunc create_func_{ nullptr };
  OpImplKernelRegistry::CompileInfoDeleterFunc delete_func_{ nullptr };
};
}  // namespace gert
#endif // GE_RUNTIME_TILING_PARSE_CONTEXT_BUILDER_H_
