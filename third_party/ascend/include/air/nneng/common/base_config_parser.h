/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_BASE_CONFIG_PARSER_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_BASE_CONFIG_PARSER_

#include <map>
#include <memory>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class BaseConfigParser;
using BaseConfigParserPtr = std::shared_ptr<BaseConfigParser>;
class BaseConfigParser {
public:
  BaseConfigParser() {}
  virtual ~BaseConfigParser() {}

  virtual Status InitializeFromOptions(const std::map<std::string, std::string> &options) {
    (void)options;
    return SUCCESS;
  }
  virtual Status InitializeFromContext() {
    return SUCCESS;
  }
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_BASE_CONFIG_PARSER_
