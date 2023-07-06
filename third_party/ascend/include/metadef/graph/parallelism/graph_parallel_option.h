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

#ifndef METADEF_INC_GRAPH_PARALLELISM_GRAPH_PARALLEL_OPTION_H_
#define METADEF_INC_GRAPH_PARALLELISM_GRAPH_PARALLEL_OPTION_H_

#include <cstdint>
#include <string>

namespace ge {
struct PipelineParallelOption {
  bool is_enabled = false;
  bool is_auto = false;
  std::string pipeline_strategy;
  int32_t pipe_stage_num = -1;
  int32_t schedule_opt_virtual_stage_num = -1;
};

struct TensorParallelOption {
  bool is_enabled = false;
  bool is_auto = false;
  int32_t tensor_parallel_size = 4;
  int32_t inter_batch_flow_num = 1;
};

struct DataParallelOption {
  bool is_enabled = false;
  bool is_auto = false;
  bool optimizer_state_sharding = false;
  bool gradient_sharding = false;
  bool model_weight_sharding = false;
  bool model_weight_prefetch = true;
  int32_t data_parallel_size = -1;
  // model weight prefetch buffer size(MB)
  uint32_t model_weight_prefetch_buffer_size = 0U;
};

struct OptimizerOffloadGraphOption {
  bool is_enabled = false;
  std::string offload; // cpu or NVME, NVME is reserved
  std::string offload_path; // NVME path, reserved
};

struct GraphParallelOption {
  int32_t graph_id = -1;
  int32_t version = -1;
  bool auto_deploy = false;
  int32_t global_batch_size = -1;
  DataParallelOption data_parallel_option;
  TensorParallelOption tensor_parallel_option;
  PipelineParallelOption pipeline_parallel_option;
  OptimizerOffloadGraphOption optimizer_offload_option;
};
}  // namespace ge

#endif  // METADEF_INC_GRAPH_PARALLELISM_GRAPH_PARALLEL_OPTION_H_
