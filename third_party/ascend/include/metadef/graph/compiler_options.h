/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#ifndef INC_GRAPH_COMPILER_OPTIONS_H_
#define INC_GRAPH_COMPILER_OPTIONS_H_

#ifdef __GNUC__
#define METADEF_ATTRIBUTE_UNUSED __attribute__((unused))
#define METADEF_FUNCTION_IDENTIFIER __PRETTY_FUNCTION__
#define METADEF_BUILTIN_PREFETCH(args_addr) __builtin_prefetch(args_addr)

#ifdef HOST_VISIBILITY
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif

#ifdef DEV_VISIBILITY
#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_DEV_VISIBILITY
#endif

#else // WINDOWS
#define METADEF_ATTRIBUTE_UNUSED
#define METADEF_FUNCTION_IDENTIFIER __FUNCSIG__
#define METADEF_BUILTIN_PREFETCH(args_addr)
#define GE_FUNC_HOST_VISIBILITY
#define GE_FUNC_DEV_VISIBILITY
#endif

#endif  // INC_GRAPH_COMPILER_OPTIONS_H_