/*
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef INC_REGISTER_OP_TILING_H
#define INC_REGISTER_OP_TILING_H

#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "register/op_tiling_registry.h"

namespace optiling {
extern "C" ge::graphStatus OpParaCalculateV2(const ge::Operator &op, OpRunInfoV2 &run_info);
extern "C" ge::graphStatus OpAtomicCalculateV2(const ge::Node &node, OpRunInfoV2 &run_info);
extern "C" ge::graphStatus OpFftsPlusCalculate(const ge::Operator &op, std::vector<OpRunInfoV2> &op_run_info);
}  // namespace optiling
#endif  // INC_REGISTER_OP_TILING_H
