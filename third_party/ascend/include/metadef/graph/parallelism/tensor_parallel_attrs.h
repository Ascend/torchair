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

#ifndef METADEF_INC_GRAPH_PARALLELISM_TENSOR_PARALLEL_ATTRS_H_
#define METADEF_INC_GRAPH_PARALLELISM_TENSOR_PARALLEL_ATTRS_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "ge/ge_api_types.h"

namespace ge {
namespace tp {
constexpr const char_t *kCommTaskTypeConcat = "Concat";
constexpr const char_t *kCommTaskTypeUniqueConcat = "UniqueConcat";
constexpr const char_t *kCommTaskTypeModifyValue = "ModifyValue";
constexpr const char_t *kCommTaskTypeSlice = "Slice";
constexpr const char_t *kCommTaskTypeSliceByAxis = "SliceByAxis";
constexpr const char_t *kCommTaskTypeSplit = "Split";
constexpr const char_t *kCommTaskTypeTranspose = "Transpose";
constexpr const char_t *kCommTaskTypeHcomAllGather = "HcomAllGather";
constexpr const char_t *kCommTaskTypeHcomAllReduce = "HcomAllReduce";
constexpr const char_t *kCommTaskTypeHcomAllReduceMean = "HcomAllReduceMean";
constexpr const char_t *kCommTaskTypeHcomReduceScatter = "HcomReduceScatter";
constexpr const char_t *kCommTaskTypeHcomBroadcast = "HcomBroadcast";
constexpr const char_t *kCommTaskTypeHcomAllToAll = "HcomAllToAll";
constexpr const char_t *kCommTaskTypeSendReceive = "SendReceive";
constexpr const char_t *kCommTaskTypeLocalReduce = "LocalReduce";
constexpr const char_t *kGraphSlicingSuffix = "_by_graph_slice_";

// tensor deployment attrs
struct DimSlice {
  int64_t begin;
  int64_t end;
};

struct DeviceIndex {
  std::string engine_type;
  std::vector<int32_t> indices;
  std::string DebugString() const;
  std::string DeviceIdToString() const;
};

bool operator==(const DeviceIndex &lhs, const DeviceIndex &rhs);
bool operator!=(const DeviceIndex &lhs, const DeviceIndex &rhs);
bool operator<(const DeviceIndex &lhs, const DeviceIndex &rhs);

struct TensorSliceDeployment {
  std::vector<std::vector<DimSlice>> axis_slices;
  std::vector<std::vector<DeviceIndex>> device_indices_each_slice;
  std::string reduce_type;
};

struct TensorDeployment {
  TensorSliceDeployment shard_deployment;
  std::string verbose;
};

struct NodeDeployment {
  std::vector<DeviceIndex> devices;
};

// P2P communications
struct CommPair {
  DeviceIndex src_device_index;
  DeviceIndex dst_device_index;
};

struct SendRecvReshardTask {
  std::vector<CommPair> comm_pairs;
  std::string parallel_group;
};

// group communications
struct CommGroup {
  std::vector<DeviceIndex> device_indices;
};

struct AllToAllReshardTask {
  std::vector<CommGroup> comm_groups;
  std::string parallel_group;
};

struct AllGatherReshardTask {
  std::vector<CommGroup> comm_groups;
  int32_t axis;  // axis to concat
  std::string parallel_group;
  std::string output_allocator;
};

struct AllReduceReshardTask {
  std::string reduction;
  std::vector<CommGroup> comm_groups;
  std::string parallel_group;
};

struct AllReduceMeanReshardTask {
  std::vector<CommGroup> comm_groups;
  int32_t axis;
  int32_t value;
  std::string parallel_group;
};

struct ReduceScatterReshardTask {
  std::string reduction;
  std::vector<CommGroup> comm_groups;
  std::string parallel_group;
};

struct BroadcastReshardTask {
  std::vector<DeviceIndex> root_device_indices;  // size == num_groups
  std::vector<CommGroup> comm_groups;
  std::string parallel_group;
};

// local reshardings
struct SliceReshardTask {
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
};

struct SliceByAxisReshardTask {
  // key: axis to split
  // value: index: slice index
  //        element: devices to deploy
  std::map<int32_t, std::vector<std::vector<DeviceIndex>>> axis_to_slice_deployments;
};

struct SplitReshardTask {
  int32_t split_dim = 0;
  int32_t num_split = 0;
};

struct ConcatReshardTask {
  int32_t concat_dim = 0;
};

struct UniqueConcatReshardTask {
  std::string unique_id;
  int32_t concat_dim = 0;
  std::vector<DeviceIndex> src_device_indices;
  DeviceIndex dst_device_index;
};

struct TransposeReshardTask {
  std::vector<int32_t> perm;
};

struct ModifyValueReshardTask {
  std::string op_type;  // mul, div
  std::vector<int64_t> value;
};

struct LocalReduceReshardTask {
  std::string op_type;
};

struct CommTask {
  std::string task_type;
  std::shared_ptr<SendRecvReshardTask> send_recv_reshard_task;
  std::shared_ptr<AllGatherReshardTask> all_gather_reshard_task;
  std::shared_ptr<AllToAllReshardTask> all_to_all_reshard_task;
  std::shared_ptr<AllReduceReshardTask> all_reduce_reshard_task;
  std::shared_ptr<AllReduceMeanReshardTask> all_reduce_mean_reshard_task;
  std::shared_ptr<ReduceScatterReshardTask> reduce_scatter_reshard_task;
  std::shared_ptr<BroadcastReshardTask> broadcast_reshard_task;
  std::shared_ptr<SplitReshardTask> split_reshard_task;
  std::shared_ptr<ConcatReshardTask> concat_reshard_task;
  std::shared_ptr<UniqueConcatReshardTask> unique_concat_reshard_task;
  std::shared_ptr<SliceReshardTask> slice_reshard_task;
  std::shared_ptr<SliceByAxisReshardTask> slice_by_axis_reshard_task;
  std::shared_ptr<TransposeReshardTask> transpose_reshard_task;
  std::shared_ptr<ModifyValueReshardTask> modify_value_reshard_task;
  std::shared_ptr<LocalReduceReshardTask> local_reduce_reshard_task;
};

struct CommStepInput {
  int32_t step_id = -1;
  int32_t output_index = -1;
};

bool operator==(const CommStepInput &lhs, const CommStepInput &rhs);
bool operator<(const CommStepInput &lhs, const CommStepInput &rhs);

struct CommStep {
  int32_t id;
  std::vector<CommStepInput> inputs;
  CommTask comm_task;
};

struct PeerInput {
  int32_t step_id = -1;
  std::string node_name;
  uint32_t input_index;
};

// reshard ops for one output tensor
struct OutputReshardRes {
  std::vector<CommStep> comm_steps;
  std::vector<PeerInput> peer_inputs;
  std::vector<DeviceIndex> device_indices;
};

struct ReshardAttr {
  std::vector<std::vector<OutputReshardRes>> reshard_infos;  // indexed by output index
};

struct SrcNodeInfo {
  int32_t inserted_node_id = -1;
  int32_t output_index = -1;
};
bool operator==(const SrcNodeInfo &lhs, const SrcNodeInfo &rhs);
bool operator<(const SrcNodeInfo &lhs, const SrcNodeInfo &rhs);

struct OrigNodeInfo {
  std::string node_name;
  int32_t sliced_id = -1;

  std::string Name() const {
    return (sliced_id == -1) ? node_name : (node_name + kGraphSlicingSuffix + std::to_string(sliced_id));
  }
};

bool operator==(const OrigNodeInfo &lhs, const OrigNodeInfo &rhs);
bool operator<(const OrigNodeInfo &lhs, const OrigNodeInfo &rhs);

struct DstNodeInfo {
  OrigNodeInfo orig_node_info;
  std::vector<uint32_t> input_indexes;

  std::string InputIndexesToString() const {
    std::string res;
    for (const uint32_t input_index : input_indexes) {
      res += std::to_string(input_index) + " ";
    }
    return res;
  }
};

bool operator==(const DstNodeInfo &lhs, const DstNodeInfo &rhs);
bool operator<(const DstNodeInfo &lhs, const DstNodeInfo &rhs);

struct InsertedNodeInput {
  SrcNodeInfo input_info;
  OrigNodeInfo orig_node_info;
};

bool operator==(const InsertedNodeInput &lhs, const InsertedNodeInput &rhs);
bool operator<(const InsertedNodeInput &lhs, const InsertedNodeInput &rhs);

struct PeerOutNodeInfo {
  SrcNodeInfo input_info;
  DstNodeInfo node_info;
};

bool operator==(const PeerOutNodeInfo &lhs, const PeerOutNodeInfo &rhs);
bool operator<(const PeerOutNodeInfo &lhs, const PeerOutNodeInfo &rhs);

struct InsertedNodeInfo {
  uint32_t id;
  CommTask task;
  std::vector<InsertedNodeInput> inputs;
};

struct OutputSlicedRes {
  std::vector<InsertedNodeInfo> inserted_nodes_info;
  std::vector<PeerOutNodeInfo> peer_out_nodes;
};

struct SlicedEdgeInfo {
  std::vector<OutputSlicedRes> steps_sliced;
};

struct TensorShapeSlicedInfo {
  std::vector<std::vector<DimSlice>> axis_slices;
};

struct NodeSliceStrategy {
  std::map<uint32_t, TensorShapeSlicedInfo> input_shape_sliced_info;
  std::map<uint32_t, TensorShapeSlicedInfo> output_shape_sliced_info;

  std::map<uint32_t, SlicedEdgeInfo> outputs_sliced_edge_info;
  std::vector<std::vector<std::pair<std::string, uint32_t>>> dependencies;
  size_t size = 1U;
};

class TensorParallelAttrs {
 public:
  static Status FromJson(const std::string &json_str, DeviceIndex &device_index);
  static Status FromJson(const std::string &json_str, NodeDeployment &node_deployment);
  static Status FromJson(const std::string &json_str, TensorDeployment &tensor_deployment);
  static Status FromJson(const std::string &json_str, CommTask &comm_task);
  static Status FromJson(const std::string &json_str, CommStep &comm_step);
  static Status FromJson(const std::string &json_str, OutputReshardRes &output_reshard_res);
  static Status FromJson(const std::string &json_str, ReshardAttr &reshard_attr);

  static std::string ToJson(const NodeDeployment &node_deployment);
  static std::string ToJson(const DeviceIndex &device_index);
  static std::string ToJson(const TensorDeployment &tensor_deployment);
  static std::string ToJson(const ReshardAttr &reshard_attr);
};
}  // namespace tp
}  // namespace ge

#endif  // METADEF_INC_GRAPH_PARALLELISM_TENSOR_PARALLEL_ATTRS_H_
