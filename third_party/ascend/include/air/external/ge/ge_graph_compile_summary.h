/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GE_GE_GRAPH_COMPILE_SUMMARY_H
#define INC_EXTERNAL_GE_GE_GRAPH_COMPILE_SUMMARY_H

#include <memory>
#include <vector>
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_feature_memory.h"
#include "graph/gnode.h"

namespace ge {
class StreamAllocationSummary;

class GE_FUNC_VISIBILITY CompiledGraphSummary {
 public:
  class Builder;
  class SummaryData;

  ~CompiledGraphSummary();
  CompiledGraphSummary &operator=(const CompiledGraphSummary &) & = delete;
  CompiledGraphSummary(const CompiledGraphSummary &) = delete;

  ///
  /// @brief get whether or not the graph is static compiled
  /// @return return true if static
  ///
  bool IsStatic() const;

  ///
  /// @brief get const memory size after compiled
  /// @param [out] size const memory size
  /// @return Status result of function
  ///
  Status GetConstMemorySize(size_t &size) const;

  ///
  /// @brief get fearturemap memory size after compiled, without input and output
  /// @param [out] size fearturemap memory size
  /// @return Status result of function
  ///
  Status GetFeatureMemorySize(size_t &size) const;

  ///
  /// @brief get fix feature memory size after compiled
  /// @param [out] size const memory size
  /// @return Status result of function
  ///
  Status GetFixedFeatureMemorySize(size_t &size) const;

  ///
  /// @brief get all type feature memory size after compiled
  /// @return vector of FeatureMemory pointer
  ///
  std::vector<FeatureMemoryPtr> GetAllFeatureMemoryTypeSize() const;

  ///
  /// @brief get refreshable fearturemap memory size after compiled, without input and output and fix memory
  /// @param [out] size fearturemap memory size
  /// @return Status result of function
  ///
  Status GetRefreshableFeatureMemorySize(size_t &size) const;

  ///
  /// @brief get whether or not the graph support featuremap memory base refreshable
  /// @param [out] v refreshable or not
  /// @return Status result of function
  ///
  Status GetFeatureMemoryBaseRefreshable (bool &v) const;

  ///
  /// @brief get the used stream number of the whole compiled graph
  /// @param [out] num used stream number
  /// @return Status result of function
  ///
  Status GetStreamNum(size_t &num) const;

  ///
  /// @brief get the used event number of the whole compiled graph
  /// @param [out] num used event number
  /// @return Status result of function
  ///
  Status GetEventNum(size_t &num) const;

  ///
  /// @brief get the output tensor shapes of the whole compiled graph
  /// @param [out] shapes vector of ge::Shape
  /// @return Status result of function
  ///
  Status GetOutputShapes(std::vector<ge::Shape> &shapes) const;

  Status GetOutputDtypes(std::vector<ge::DataType> &dtypes) const;

  Status GetIOIndexesWithSameAddr(std::vector<std::pair<uint32_t, uint32_t>> &io_indexes) const;

  Status GetInputShardMethod(std::map<std::string, std::map<int32_t, std::vector<std::pair<int64_t, int64_t>>>>
                             &device_id_to_tensor_deployment) const;
  
  ///
  /// @brief get StreamAllocationSummary which has stream infos
  /// @param [out] StreamAllocationSummary
  /// @return Status result of function
  ///
  Status GetStreamAllocationSummary(std::shared_ptr<StreamAllocationSummary> &stream_allocation) const;

 private:
  CompiledGraphSummary() = default;
  std::shared_ptr<SummaryData> data_{nullptr};
};

using CompiledGraphSummaryPtr = std::shared_ptr<CompiledGraphSummary>;

class LogicalStreamAllocationInfo;
class GE_FUNC_VISIBILITY StreamAllocationSummary {
public:
  class StreamAllocationSummaryImpl;
  StreamAllocationSummary();
  ~StreamAllocationSummary();

  /**
   * @brief get root path and subgraph's AscendString of the stream graph in DOT file format
   * @return map, key is graph name, value is stream graph in DOT file format
   */
  const std::map<AscendString, AscendString> &ToStreamGraph() const;

  /**
   * @brief get root path and subgraph's logical stream infos
   * @return map, key is graph name, value is LogicalStreamAllocationInfo
   */
  const std::map<AscendString, std::vector<LogicalStreamAllocationInfo>> &GetAllLogicalStreamInfos() const;

private:
  friend class CompiledGraphSummary::Builder;
  std::unique_ptr<StreamAllocationSummaryImpl> impl_;
};

class GE_FUNC_VISIBILITY LogicalStreamAllocationInfo {
public:
  class LogicalStreamAllocationInfoImpl;

  LogicalStreamAllocationInfo();
  ~LogicalStreamAllocationInfo();
  LogicalStreamAllocationInfo(const LogicalStreamAllocationInfo &stream_info);
  LogicalStreamAllocationInfo &operator=(const LogicalStreamAllocationInfo &stream_info);

  /**
   * @brief get info of current logical stream
   * @return AscendString, eg: "logic_stream_id: 0, user_stream_label: 11, is_assigned_by_user_stream_pass: false,
   * attached_stream_ids: , physical_model_stream_num: 1, hccl_followed_stream_num: 0.";
   */
  AscendString ToStringInfo() const;

  /**
   * @brief get stream id of current logical stream
   * @return int64_t, logical stream id
   */
  int64_t GetLogicalStreamId() const;

  /**
   * @brief get UserStreamLabel of current logical stream
   * @return AscendString, UserStreamlabel
   */
  AscendString GetUsrStreamLabel() const;

  /**
   * @brief current logical stream is assigned by user register stream pass
   * @return bool, true or false
   */
  bool IsAssignedByStreampass() const;

  /**
   * @brief get logical attached stream ids of current logical stream
   * @return vector of int64_t, logical attached stream ids
   */
  std::vector<int64_t> GetAttachedStreamIds() const;

  /**
   * @brief get physical model stream num of current logical stream
   * @return size_t, physical stream num
   */
  size_t GetPhysicalStreamNum() const;

  /**
   * @brief get hccl followed stream num of current logical stream
   * @return size_t, hccl followed stream num
   */
  size_t GetHcclFollowedStreamNum() const;

  /**
   * @brief get all GNode on current logical stream
   * @return vector of GNode
   */
  const std::vector<GNode> &GetAllGNodes() const;

  private:
    friend class StreamAllocationSummary::StreamAllocationSummaryImpl;
    std::unique_ptr<LogicalStreamAllocationInfoImpl> impl_;
};

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief get root graph and subgraph's logical stream info's string
 * @param [in] compiled_graph_summary
 * @param [out] map, key is graph name, value is vector of logical stream info's string which index is logical stream id
 * @return result of function
 */
ge::Status GEStreamAllocationSummaryGetStringInfos(const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<AscendString>> &graph_to_string_infos);

/**
 * @brief get root path and subgraph's logical stream ids
 * @param [in] compiled_graph_summary
 * @param [out] map, key is graph name, value is vector of logical stream id which index is logical stream id
 * @return result of function
 */
ge::Status GEStreamAllocationSummaryGetLogicalStreamIds(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<int64_t>> &graph_to_logical_stream_ids);

/**
 * @brief get root path and subgraph's user stream label
 * @param [in] compiled_graph_summary
 * @param [out] map, key is graph name, value is vector of User Stream Label which index is logical stream id
 * @return result of function
 */
ge::Status GEStreamAllocationSummaryGetUsrStreamLabels(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<AscendString>> &graph_to_user_stream_labels);

/**
 * @brief get root path and subgraph's is assigned by stream pass
 * @param [in] compiled_graph_summary
 * @param [out] map, key is graph name, value is vector of is assigned by stream pass which index is logical stream id
 * @return result of function 
 */
ge::Status GEStreamAllocationSummaryGetIsAssignedByStreampass(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<bool>> &graph_to_is_assigned_by_stream_pass);

/**
 * @brief get root path and subgraph's attached stream ids
 * @param [in] compiled_graph_summary
 * @param [out] map, key is graph name, value is vector of attached stream ids which index is logical stream id
 * @return result of function
 */
ge::Status GEStreamAllocationSummaryGetAttachedStreamIds(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<std::vector<int64_t>>> &graph_to_attached_stream_ids);

/**
 * @brief get root path and subgraph's physical stream nums
 * @param [in] compiled_graph_summary
 * @param [out] map, key is graph name, value is vector of physical stream nums which index is logical stream id
 * @return result of function
 */
ge::Status GEStreamAllocationSummaryGetPhysicalStreamNums(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<size_t>> &graph_to_physical_stream_nums);

/**
 * @brief get root path and subgraph's hccl followed stream nums
 * @param [in] compiled_graph_summary
 * @param [out] map, key is graph name, value is vector of hccl followed stream nums which index is logical stream id
 * @return result of function
 */
ge::Status GEStreamAllocationSummaryGetHcclFollowedStreamNums(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<int64_t>> &graph_to_hccl_followed_stream_nums);

/**
 * @brief get root path and subgraph's all nodes
 * @param [in] compiled_graph_summary
 * @param [out] map, key is graph name, value is vector of vector of all nodes which index is logical stream id
 * @return result of function
 */
ge::Status GEStreamAllocationSummaryGetAllGNodes(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, std::vector<std::vector<GNode>>> &graph_to_all_nodes);

/**
 * @brief get root path and subgraph's stream graph
 * @param [in] compiled_graph_summary
 * @param [out] map, key is graph name, value is string of stream graph
 * @return result of function
 */
ge::Status GEStreamAllocationSummaryGetStreamGraphs(
    const ge::CompiledGraphSummary &compiled_graph_summary,
    std::map<AscendString, AscendString> &graph_to_stream_graphs);

#ifdef __cplusplus
}
#endif
}  // namespace ge
#endif  // INC_EXTERNAL_GE_GE_GRAPH_COMPILE_SUMMARY_H
