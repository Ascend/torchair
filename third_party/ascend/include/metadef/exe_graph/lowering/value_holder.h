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

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_H_
#include <cstdint>
#include <string>
#include <memory>
#include <atomic>

#include "graph/buffer.h"
#include "graph/any_value.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "common/hyper_status.h"
#include "graph_frame.h"
#include "exe_graph/runtime/tensor.h"

namespace gert {
namespace bg {
class ValueHolder;
using ValueHolderPtr = std::shared_ptr<ValueHolder>;
class ValueHolder {
 public:
  enum class ValueHolderType {
    kConst,      // 常量，执行时不变
    kFeed,       // 执行时外部指定
    kOutput,     // 由node产生，包含数据输出与控制输出
    kConstData,  // 常量Const，执行时由外部指定，执行时不变
    // Add new type definitions here
    kValueHolderTypeEnd
  };

  using NodeHolder = ge::Node;
  using NodeHolderPtr = ge::NodePtr;
  using GraphHolder = ge::ComputeGraph;
  using GraphHolderPtr = ge::ComputeGraphPtr;
  using ExecuteGraphPtr = ge::ComputeGraphPtr;

  class CurrentComputeNodeGuarder {
   public:
    explicit CurrentComputeNodeGuarder(ge::NodePtr old_node) : old_node_(std::move(old_node)) {}
    ~CurrentComputeNodeGuarder() {
      ValueHolder::SetCurrentComputeNode(old_node_);
    }

   private:
    ge::NodePtr old_node_;
  };

  ValueHolder(const ValueHolder &other) = delete;
  ValueHolder &operator=(const ValueHolder &other) = delete;
  ~ValueHolder();

  bool IsOk() const noexcept;

  int64_t GetId() const noexcept;
  ValueHolderType GetType() const noexcept;
  const NodeHolder *GetNode() const noexcept;
  const GraphHolder *GetGraph() const noexcept;
  ValueHolderPtr GetGuarder() const noexcept;

  int32_t GetOutIndex() const noexcept;
  // ref-from other的含义是，本value指向了other（本value没有独立的内存）
  ge::graphStatus RefFrom(const ValueHolderPtr &other);

  // 在other产生后，本holder的生命周期才结束
  void ReleaseAfter(const ValueHolderPtr &other);

  const int32_t &GetPlacement() const;
  void SetPlacement(const int32_t &placement);

  std::vector<ValueHolderPtr> AppendOutputs(size_t append_count);

  static ValueHolderPtr CreateError(const ge::char_t *fmt, ...);
  static ValueHolderPtr CreateError(const ge::char_t *fmt, va_list arg);
  static ValueHolderPtr CreateConst(const void *data, size_t size, bool is_string = false);
  static ValueHolderPtr CreateFeed(int64_t index);
  static ValueHolderPtr CreateConstData(int64_t index);

  static ValueHolderPtr CreateSingleDataOutput(const ge::char_t *node_type, const std::vector<ValueHolderPtr> &inputs);
  static std::vector<ValueHolderPtr> CreateDataOutput(const ge::char_t *node_type,
                                                      const std::vector<ValueHolderPtr> &inputs, size_t out_count);
  static ValueHolderPtr CreateVoid(const ge::char_t *node_type, const std::vector<ValueHolderPtr> &inputs);
  static ValueHolderPtr CreateVoidGuarder(const ge::char_t *node_type, const ValueHolderPtr &resource,
                                          const std::vector<ValueHolderPtr> &args);
  static HyperStatus AddDependency(const ValueHolderPtr &src, const ValueHolderPtr &dst);

  /**
   * 压栈一个Root GraphFrame，只有栈底的GraphFrame才被称为ROOT GraphFrame，因此调用此借口前，需要保证栈内不存在GraphFrame，否则会失败
   * @return 成功后，返回创建好的GraphFrame指针，失败时返回空指针
   */
  static GraphFrame *PushGraphFrame();
  /**
   * 压栈一个非root的GraphFrame
   * @param belongs 新加入的GraphFrame所归属的ValueHolder，新压栈的GraphFrame会被挂在该ValueHolder所归属的Node上
   * @param graph_name 挂接GraphFrame到Node时，使用的name
   * @return 创建且挂接成功后，返回创建好的GraphFrame指针，失败时返回空指针
   */
  static GraphFrame *PushGraphFrame(const ValueHolderPtr &belongs, const ge::char_t *graph_name);
  static std::unique_ptr<GraphFrame> PopGraphFrame();
  static std::unique_ptr<GraphFrame> PopGraphFrame(const std::vector<ValueHolderPtr> &outputs,
                                                   const std::vector<ValueHolderPtr> &targets);
  static std::unique_ptr<GraphFrame> PopGraphFrame(const std::vector<ValueHolderPtr> &outputs,
                                                   const std::vector<ValueHolderPtr> &targets,
                                                   const ge::char_t *out_node_type);
  static GraphFrame *GetCurrentFrame();
  static GraphHolder *GetCurrentGraph();
  static void SetCurrentComputeNode(const ge::NodePtr &node);
  static void AddRelevantInputNode(const ge::NodePtr &node);
  static std::unique_ptr<CurrentComputeNodeGuarder> SetScopedCurrentComputeNode(const ge::NodePtr &node);

  static NodeHolderPtr AddNode(const ge::char_t *node_type, size_t input_count,
    size_t output_count, const GraphFrame &frame);
  static std::vector<ValueHolderPtr> CreateFromNode(const NodeHolderPtr &node,
    size_t start_index, size_t create_count);
  static ValueHolderPtr CreateFromNode(NodeHolderPtr node, int32_t index, ValueHolderType type);
  static std::string GenerateNodeName(const ge::char_t *node_type, const GraphFrame &frame);

  static std::vector<ValueHolderPtr> GetLastExecNodes();

 private:
  ValueHolder();
  static std::vector<ValueHolderPtr> CreateFromNode(const NodeHolderPtr &node, size_t out_count);
  static NodeHolderPtr CreateNode(const ge::char_t *node_type, const std::vector<ValueHolderPtr> &inputs,
                                  size_t out_count);

 private:
  static std::atomic<int64_t> id_generator_;
  int64_t id_;
  ValueHolder::ValueHolderType type_;
  GraphHolderPtr graph_;  // no use, to be deleted
  ge::NodePtr node_;
  int32_t index_;
  int32_t placement_;
  std::unique_ptr<char[]> error_msg_;
  ValueHolderPtr guarder_;
};
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_H_
