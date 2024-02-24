/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef INC_COMMON_OPSKERNEL_OPS_KERNEL_BUILDER_H_
#define INC_COMMON_OPSKERNEL_OPS_KERNEL_BUILDER_H_

#include "external/ge_common/ge_api_error_codes.h"
#include "cce/aicpu_engine_struct.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/node.h"
#include "external/ge_common/ge_api_types.h"
#include "proto/task.pb.h"

namespace ge {
class OpsKernelBuilder {
 public:
  enum class Mode : uint32_t {
    kNormal,
    kFfts,
    kFftsPlus
  };
  OpsKernelBuilder() = default;
  virtual ~OpsKernelBuilder() = default;
  OpsKernelBuilder(const OpsKernelBuilder &) = delete;
  OpsKernelBuilder(OpsKernelBuilder &&) = delete;
  OpsKernelBuilder &operator=(const OpsKernelBuilder &)& = delete;
  OpsKernelBuilder &operator=(OpsKernelBuilder &&)& = delete;

  // initialize OpsKernelBuilder
  virtual Status Initialize(const std::map<std::string, std::string> &options) = 0;

  // finalize OpsKernelBuilder
  virtual Status Finalize() = 0;

  // memory allocation requirement
  virtual Status CalcOpRunningParam(Node &node) = 0;

  // generate task for op
  virtual Status GenerateTask(const Node &node, RunContext &context,
                              std::vector<domi::TaskDef> &tasks) = 0;

  // generate task for op with different mode
  virtual Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks,
                              OpsKernelBuilder::Mode) {
    (void)node;
    (void)context;
    (void)tasks;
    return SUCCESS;
  }

  // only call aicpu interface to generate task struct
  virtual Status GenSingleOpRunTask(const NodePtr &node, STR_FWK_OP_KERNEL &task, std::string &task_info) {
    (void)node;
    (void)task;
    (void)task_info;
    return FAILED;
  }

  // only call aicpu interface to generate task struct
  virtual Status GenMemCopyTask(const uint64_t count, STR_FWK_OP_KERNEL &task, std::string &task_info) {
    (void)count;
    (void)task;
    (void)task_info;
    return FAILED;
  }
};
}  // namespace ge
#endif // INC_COMMON_OPSKERNEL_OPS_KERNEL_BUILDER_H_
