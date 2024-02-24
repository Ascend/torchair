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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_PASS_DESC_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_PASS_DESC_H_
#include <stdint.h>
namespace fe {
using PassAttr = uint64_t;
const PassAttr FORBIDDEN_CLOSE = 0x01UL;  // forbidden close, can not be closed by fusion switch
const PassAttr NEED_SORT = 0x02UL;  // need topological sorting before executing
const PassAttr SINGLE_SCENE_OPEN = 0x04UL;  // open for single op scene, can be close by fusion switch
const PassAttr FE_PASS = 0x08UL;  // graph passes and ub passes in air project
const PassAttr ENABLE_AUTO_FUSION = 0x10UL;  // whether using auto match fusion frame
const PassAttr ALWAYS_GENERALIZE = 0x20UL;
const PassAttr PRUNING = 0x40UL; // whether the pass need to pre run before graph fusion
const PassAttr ENABLE_FUSION_CHECK = 0x80UL; // enable do fusion check for matched fusion nodes during ub fusion
const PassAttr PASS_BIT_MASK = 0x1UL;  // check if the loweset bit of pass is 1

enum class PassAttrType {
  FRBDN_CLOSE = 0, // Mark those passes that cannot be turned off in graph mode
  NEED_TOPO_SORT = 1, // Mark those graph fusion passes that need topological sorting before executing
  SINGLE_OP_SCENE_MUST_ON = 2, // Mark those passes that must be turned on in single-op mode or jit_compile=false
  FE_PASS_FLAG = 3, // Mark those passes that belong to FE
  AUTO_FUSION_FLAG = 4, // Using auto match fusion frame
  /* The OpDescs in the patterns of this kind fusion pass are able to be generalized in all scenarios.
   * For example, they can ignore the value dependency restrict. */
  ALWAYS_GENERALIZE_FLAG = 5,
  PRUNING_FLAG = 6,
  FUSION_CHECK_FLAG = 7  // Do fusion check for matched fusion nodes during ub fusion
};

bool IsPassAttrTypeOn(PassAttr pass_attr, PassAttrType attr_type);
}  // namespace fe
#endif  // INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_PASS_DESC_H_