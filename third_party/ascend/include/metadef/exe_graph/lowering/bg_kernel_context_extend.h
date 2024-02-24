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

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_KERNEL_CONTEXT_EXTEND_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_KERNEL_CONTEXT_EXTEND_H_
#include "graph/node.h"
#include "buffer_pool.h"
#include "external/register/op_impl_kernel_registry.h"
namespace gert {
namespace bg {
std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool);
std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool, size_t &total_size);
std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool,
    const gert::OpImplKernelRegistry::PrivateAttrList &private_attrs, size_t &total_size);
std::unique_ptr<uint8_t[]> CreateComputeNodeInfoWithoutIrAttr(const ge::NodePtr &node, BufferPool &buffer_pool,
    const gert::OpImplKernelRegistry::PrivateAttrList &private_attrs, size_t &total_size);
}
}
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_KERNEL_CONTEXT_EXTEND_H_
