/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

/*!
 * \file cache_management_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CACHE_MANAGEMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CACHE_MANAGEMENT_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 *@brief Operators for managing cache memory.

 *@par Inputs:
 *: A tensor of the type TensorType::NumberType().

 *@par Attributes:
 *@li max_size: The maximum memory size required for operation.
 *@li type: An optional int. The default value is '6', indicating prefetch.\n
 */
REG_OP(Cmo)
    .INPUT(src, TensorType::NumberType())
    .REQUIRED_ATTR(max_size, Int)
    .ATTR(type, Int, 6) // 6:prefetch
    .ATTR(offset, Int, 0)
    .OP_END_FACTORY_REG(Cmo)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_CACHE_MANAGEMENT_OPS_H_