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
namespace ascir {
namespace cg {
namespace {
thread_local CgContext *t_context = nullptr;
}

CgContext *CgContext::GetThreadLocalContext() {
  return t_context;
}
void CgContext::SetThreadLocalContext(CgContext *context) {
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
const std::vector<AxisId> &CgContext::GetLoopAxisIds() const {
  return loop_axis_ids_cache_;
}
LoopGuard::~LoopGuard() {
  CgContext::SetThreadLocalContext(nullptr);
}
LoopGuard::LoopGuard(std::vector<Axis> axes) {
  CgContext::SetThreadLocalContext(&context);
  context.SetLoopAxes(std::move(axes));
}
}  // namespace cg
}  // namespace ascir