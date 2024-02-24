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

#ifndef INC_COMMON_OPSKERNEL_OPS_KERNEL_INFO_STORE_H_
#define INC_COMMON_OPSKERNEL_OPS_KERNEL_INFO_STORE_H_

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <climits>

#include "common/opskernel/ge_task_info.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/ge_common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "external/graph/operator.h"

namespace ge {
class OpsKernelInfoStore {
 public:
  OpsKernelInfoStore() = default;

  virtual ~OpsKernelInfoStore() = default;
  OpsKernelInfoStore(const OpsKernelInfoStore &) = delete;
  OpsKernelInfoStore(OpsKernelInfoStore &&) = delete;
  OpsKernelInfoStore &operator=(const OpsKernelInfoStore &)& = delete;
  OpsKernelInfoStore &operator=(OpsKernelInfoStore &&)& = delete;

  // initialize opsKernelInfoStore
  virtual Status Initialize(const std::map<std::string, std::string> &options) = 0;

  // close opsKernelInfoStore
  virtual Status Finalize() = 0; /*lint -e148*/

  virtual Status CreateSession(const std::map<std::string, std::string> &session_options) {
    (void)session_options;
    return SUCCESS;
  }

  virtual Status DestroySession(const std::map<std::string, std::string> &session_options) {
    (void)session_options;
    return SUCCESS;
  }

  // get all opsKernelInfo
  virtual void GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const = 0;

  // whether the opsKernelInfoStore is supported based on the operator attribute
  virtual bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const = 0;

  virtual bool CheckAccuracySupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason,
                                      const bool realQuery = false) const {
    (void)realQuery;
    return CheckSupported(opDescPtr, un_supported_reason);
  }
  // opsFlag opsFlag[0] indicates constant folding is supported or not
  virtual void opsFlagCheck(const ge::Node &node, std::string &opsFlag) {
    (void)node;
    (void)opsFlag;
  };

  // only call fe engine interface to compile single op
  virtual Status CompileOp(std::vector<ge::NodePtr> &node_vec) {
    (void) node_vec;
    return SUCCESS;
  }
  virtual Status CompileOpRun(std::vector<ge::NodePtr> &node_vec) {
    (void)node_vec;
    return SUCCESS;
  }

  // prepare task for op
  virtual Status PrepareTaskAsync(GETaskInfo &task) {
    (void)task;
    return SUCCESS;
  }

  // load task for op
  virtual Status LoadTask(GETaskInfo &task) {
    (void)task;
    return SUCCESS;
  }

  virtual bool CheckSupported(const ge::NodePtr &node, std::string &un_supported_reason) const {
    if (node == nullptr) {
      return false;
    }
    return CheckSupported(node->GetOpDesc(), un_supported_reason);
  }

  virtual bool CheckAccuracySupported(const ge::NodePtr &node, std::string &un_supported_reason,
                                      const bool realQuery = false) const {
    (void)realQuery;
    if (node == nullptr) {
      return false;
    }
    return CheckAccuracySupported(node->GetOpDesc(), un_supported_reason, realQuery);
  }
  // Set cut support info
  virtual Status SetCutSupportedInfo(const ge::NodePtr &node) {
    (void)node;
    return SUCCESS;
  }
  // unload task for op
  virtual Status UnloadTask(GETaskInfo &task) {
    (void)task;
    return SUCCESS;
  }

  // fuzz compile interface
  virtual Status FuzzCompileOp(std::vector<ge::NodePtr> &node_vec) {
    (void) node_vec;
    return SUCCESS;
  }

  // Query information such as foramt/dtype/impl supported by operators (extensible)
  virtual bool GetNodeSupportInfo(const OperatorPtr &op, std::string &support_info) {
    (void)op;
    (void)support_info;
    return false;
  }

  virtual bool CheckSupported(const ge::NodePtr &node, std::string &un_supported_reason,
                              CheckSupportFlag &flag) const {
    (void)flag;
    return CheckSupported(node, un_supported_reason);
  }
};
}  // namespace ge
#endif  // INC_COMMON_OPSKERNEL_OPS_KERNEL_INFO_STORE_H_
