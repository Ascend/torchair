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

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_LOWERING_GLOBAL_DATA_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_LOWERING_GLOBAL_DATA_H_
#include <map>
#include "proto/task.pb.h"
#include "value_holder.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/allocator.h"
#include "exe_graph/runtime/execute_graph_types.h"
#include "register/op_impl_space_registry.h"
#include "exe_graph/lowering/lowering_opt.h"

namespace gert {
constexpr int64_t kDefaultMainStreamId = 0;
// todo change to get stream num from model_desc const data
constexpr const ge::char_t *kGlobalDataModelStreamNum = "ModelStreamNum";
class LoweringGlobalData {
 public:
  struct NodeCompileResult {
    const std::vector<domi::TaskDef> &GetTaskDefs() const {
      return task_defs;
    }
    std::vector<domi::TaskDef> task_defs;
  };

  std::vector<bg::ValueHolderPtr> LoweringAndSplitRtStreams(int64_t stream_num);
  bg::ValueHolderPtr GetStreamById(int64_t logic_stream_id) const;
  inline bg::ValueHolderPtr GetStream() const {
    int64_t current_stream_id = kDefaultMainStreamId;
    if ((bg::ValueHolder::GetCurrentFrame() != nullptr) &&
        (bg::ValueHolder::GetCurrentFrame()->GetCurrentComputeNode() != nullptr)) {
      current_stream_id = bg::ValueHolder::GetCurrentFrame()->GetCurrentComputeNode()->GetOpDesc()->GetStreamId();
    }
    return GetStreamById(current_stream_id);
  }

  const NodeCompileResult *FindCompiledResult(const ge::NodePtr &node) const;
  LoweringGlobalData &AddCompiledResult(const ge::NodePtr &node, NodeCompileResult compile_result);

  void *GetGraphStaticCompiledModel(const std::string &graph_name) const;
  LoweringGlobalData &AddStaticCompiledGraphModel(const std::string &graph_name, void *const model);

  bg::ValueHolderPtr GetL1Allocator(const AllocatorDesc &desc) const;
  LoweringGlobalData &SetExternalAllocator(bg::ValueHolderPtr &&allocator);
  LoweringGlobalData &SetExternalAllocator(bg::ValueHolderPtr &&allocator, const ExecuteGraphType graph_type);

  bg::ValueHolderPtr GetOrCreateL1Allocator(const AllocatorDesc desc);
  bg::ValueHolderPtr GetOrCreateL2Allocator(int64_t logic_stream_id, const AllocatorDesc desc);
  bg::ValueHolderPtr GetInitL2Allocator(const AllocatorDesc desc) const;
  bg::ValueHolderPtr GetMainL2Allocator(int64_t logic_stream_id, const AllocatorDesc desc) const;
  inline bg::ValueHolderPtr GetOrCreateAllocator(const AllocatorDesc desc) {
    int64_t current_stream_id = kDefaultMainStreamId;
    if ((bg::ValueHolder::GetCurrentFrame() != nullptr) &&
        (bg::ValueHolder::GetCurrentFrame()->GetCurrentComputeNode() != nullptr)) {
      current_stream_id = bg::ValueHolder::GetCurrentFrame()->GetCurrentComputeNode()->GetOpDesc()->GetStreamId();
    }
    return GetOrCreateL2Allocator(current_stream_id, desc);
  }
  bg::ValueHolderPtr GetOrCreateAllL2Allocators();

  bg::ValueHolderPtr GetOrCreateUniqueValueHolder(const std::string &name,
                                                  const std::function<bg::ValueHolderPtr()> &builder);
  std::vector<bg::ValueHolderPtr> GetOrCreateUniqueValueHolder(const std::string &name,
      const std::function<std::vector<bg::ValueHolderPtr>()> &builder);
  bg::ValueHolderPtr GetUniqueValueHolder(const std::string &name) const;
  void SetUniqueValueHolder(const std::string &name, const bg::ValueHolderPtr &holder);
  void SetValueHolders(const string &name, const bg::ValueHolderPtr &holder);
  size_t GetValueHoldersSize(const string &name);

  void SetModelWeightSize(const size_t require_weight_size);
  size_t GetModelWeightSize() const;
  const gert::OpImplSpaceRegistryPtr &GetSpaceRegistry() const {
    return space_registry_;
  };
  void SetSpaceRegistry(gert::OpImplSpaceRegistryPtr space_registry) {
    space_registry_ = space_registry;
  };
  const LoweringOption &GetLoweringOption() const;
  void SetLoweringOption(const LoweringOption &lowering_option);

  void SetStaicModelWsSize(const int64_t require_ws_size) {
    static_model_ws_size = require_ws_size;
  }

  int64_t GetStaticModelWsSize() const {
    return static_model_ws_size;
  }

  void SetFixedFeatureMemoryBase(const void * const memory, const size_t size) {
    fixed_feature_mem_ = std::make_pair(memory, size);
  }

  const std::pair<const void *, size_t> &GetFixedFeatureMemoryBase() const { return fixed_feature_mem_; }
 private:
  struct HolderByGraphs {
    bg::ValueHolderPtr holders[static_cast<size_t>(ExecuteGraphType::kNum)];
  };
  struct HoldersByGraphs {
    std::vector<bg::ValueHolderPtr> holders[static_cast<size_t>(ExecuteGraphType::kNum)];
  };

  bg::ValueHolderPtr GetOrCreateInitL2Allocator(const AllocatorDesc desc);
  bg::ValueHolderPtr GetExternalAllocator(const bool from_init, const string &key, const AllocatorDesc &desc);

 private:
  std::unordered_map<std::string, NodeCompileResult> node_name_to_compile_result_holders_;
  std::map<std::string, void *> graph_to_static_models_;
  std::map<std::string, std::vector<bg::ValueHolderPtr>> unique_name_to_value_holders_;
  HoldersByGraphs streams_;
  HolderByGraphs external_allocators_;
  // todo need delete and change to const_data after const_data is ready
  int64_t model_weight_size_;
  int64_t static_model_ws_size;
  OpImplSpaceRegistryPtr space_registry_;
  LoweringOption lowering_option_;
  std::pair<const void *, size_t> fixed_feature_mem_;
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_LOWERING_GLOBAL_DATA_H_
