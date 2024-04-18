/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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
#include "cg_utils.h"
#include "node_utils_ex.h"
namespace ascir {
namespace cg {
namespace {
thread_local std::weak_ptr<CgContext> t_context;

int64_t GenNextExecId(const ge::ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    throw std::invalid_argument("Invalid graph");
  }
  static const std::string kExecIdKey = "cg.ExecId";
  auto id = graph->GetExtAttr<int64_t>(kExecIdKey);
  if (id == nullptr) {
    graph->SetExtAttr(kExecIdKey, static_cast<int64_t>(1));
    return 0;
  }
  return (*id)++;
}
}  // namespace

CgContext *CgContext::GetThreadLocalContext() {
  return GetSharedThreadLocalContext().get();
}
std::shared_ptr<CgContext> CgContext::GetSharedThreadLocalContext() {
  return t_context.lock();
}
void CgContext::SetThreadLocalContext(const std::shared_ptr<CgContext> &context) {
  t_context = context;
}
const std::vector<Axis> &CgContext::GetLoopAxes() const {
  return loop_axes_;
}
void CgContext::SetLoopAxes(std::vector<Axis> axes) {
  loop_axes_ = std::move(axes);
  loop_axis_ids_cache_.clear();
  loop_axis_ids_cache_.reserve(loop_axes_.size());
  for (const auto &axis : loop_axes_) {
    loop_axis_ids_cache_.emplace_back(axis.id);
  }
}
void CgContext::PushLoopAxes(const vector<Axis> &axes) {
  for (const auto &axis : axes) {
    loop_axes_.emplace_back(axis);
    loop_axis_ids_cache_.emplace_back(axis.id);
  }
}
void CgContext::PopBackLoopAxes(const vector<Axis> &axes) {
  for (auto iter = axes.rbegin(); iter != axes.rend(); ++iter) {
    if (loop_axis_ids_cache_.empty()) {
      throw std::invalid_argument("Axes stack is empty");
    }
    auto last_id = *(loop_axis_ids_cache_.rbegin());
    if (last_id != iter->id) {
      throw std::invalid_argument("Pop Axes order unmatch");
    }
    loop_axis_ids_cache_.pop_back();
    loop_axes_.pop_back();
  }
}
const std::vector<AxisId> &CgContext::GetLoopAxisIds() const {
  return loop_axis_ids_cache_;
}
void CgContext::SetBlockLoopEnd(AxisId id) {
  block_loop_end_ = id;
}
AxisId CgContext::GetBlockLoopEnd() const {
  return block_loop_end_;
}
void CgContext::SetVectorizedLoopEnd(AxisId id) {
  vectorized_loop_end_ = id;
}
AxisId CgContext::GetVectorizedLoopEnd() const {
  return vectorized_loop_end_;
}
void CgContext::SetLoopEnd(AxisId id) {
  loop_end_ = id;
}
AxisId CgContext::GetLoopEnd() const {
  return loop_end_;
}
void CgContext::SetOption(const LoopOption &option) {
  option_ = option;
}
const LoopOption &CgContext::GetOption() const {
  return option_;
}
LoopGuard::~LoopGuard() {
  context_->PopBackLoopAxes(loop_axes_);
}
LoopGuard::LoopGuard(std::vector<Axis> axes) {
  context_ = CgContext::GetSharedThreadLocalContext();
  if (context_ == nullptr) {
    context_ = std::make_shared<CgContext>();
    CgContext::SetThreadLocalContext(context_);
  }

  loop_axes_ = std::move(axes);
  context_->PushLoopAxes(loop_axes_);
}
std::unique_ptr<LoopGuard> LoopGuard::Create(std::vector<Axis> axes, const LoopOption &option) {
  auto loop_guard = std::make_unique<LoopGuard>(std::move(axes));
  loop_guard->context_->SetOption(option);
  return loop_guard;
}

void AddToGraphFollowOp(const ge::Operator &op, ge::Operator &new_op) {
  auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  if (node == nullptr) {
    throw std::invalid_argument(
        "The input node should be created by bg::Function, "
        "the input node may be created from ops::OpType");
  }
  auto graph = node->GetOwnerComputeGraph();
  auto new_node = graph->AddNode(ge::OpDescUtils::GetOpDescFromOperator(new_op));
  auto tmp_new_op = ge::OpDescUtils::CreateOperatorFromNode(new_node);
  std::swap(tmp_new_op, new_op);
}
int64_t GenNextExecId(const ge::Operator &op) {
  auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  if (node == nullptr) {
    throw std::invalid_argument(
        "The input node should be created by bg::Function, "
        "the input node may be created from ops::OpType");
  }
  return GenNextExecId(node->GetOwnerComputeGraph());
}
int64_t GenNextExecId(const Graph &graph) {
  return GenNextExecId(ge::GraphUtilsEx::GetComputeGraph(graph));
}
}  // namespace cg
}  // namespace ascir