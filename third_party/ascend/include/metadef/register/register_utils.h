/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef REGISTER_REGISTER_UTILS_H
#define REGISTER_REGISTER_UTILS_H

#include <google/protobuf/message.h>
#include "external/register/register_types.h"
#include "external/register/register_error_codes.h"
#include "external/register/register.h"
#include "external/graph/operator.h"

namespace domi {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OperatorAutoMapping(const Message *op_src, ge::Operator &op);
}  // namespace domi
#endif  // REGISTER_REGISTER_UTILS_H
