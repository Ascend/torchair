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

#ifndef INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_
#define INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_

#include <cstdint>
#include <string>
#include <vector>
#include "runtime/rt.h"
#include "graph/op_desc.h"

namespace ge {
struct HcclDumpInfo {
  uint32_t task_id;
  uint32_t stream_id;
  uint32_t sub_task_type;
  void *input_addr;
  uint64_t input_size;
  void *output_addr;
  uint64_t output_size;
};

struct DvppInfo {
  OpDescPtr op_desc;
  std::vector<void *> io_addrs;
  uint32_t sqe[16];
};

// when need to eliminate GETaskKernelHcclInfo, so not need DAVINCI_TRAIN/DAVINCI_CLOUD
struct GETaskKernelHcclInfo {
  std::string input_name;
  std::string hccl_type;
  void *inputDataAddr;
  void *outputDataAddr;
  void *workSpaceAddr;
  int32_t count;
  int32_t dataType;
  int32_t opType;
  int64_t rootId;
  uint64_t workSpaceMemSize;
  std::vector<int64_t> dims;
  std::vector<rtStream_t> hcclStreamList;
  std::vector<HcclDumpInfo> hccl_dump_info;
  std::vector<void *> global_workspace_addr;
  uint32_t hcclQosCfg;
  std::vector<void *> inputDataAddrs;
  std::vector<void *> outputDataAddrs;
  std::vector<void *> workSpaceAddrs;
  std::vector<uint64_t> workSpaceMemSizes;
  std::vector<int32_t> inputZeroCopyFlags;
  std::vector<int32_t> outputZeroCopyFlags;
};

struct GETaskInfo {
  uint32_t id;
  uint16_t type;
  uint32_t streamID;
  void *stream;  // rtKernelLaunch input argument
  void *event;
  void *privateDef;
  uint32_t privateDefLen;
  void *opsKernelStorePtr;

  std::vector<GETaskKernelHcclInfo> kernelHcclInfo;
  DvppInfo dvpp_info;
};

struct HcomRemoteAccessAddrInfo
{
  uint32_t remotetRankID;
  uint64_t remoteAddr;  // host embedding table address
  uint64_t localAddr;  // device HBM address
  uint64_t length;   // memory Length in Bytes
};


}  // namespace ge
#endif  // INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_
