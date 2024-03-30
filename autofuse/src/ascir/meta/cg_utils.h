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

#ifndef AUTOFUSE_CG_UTILS_H
#define AUTOFUSE_CG_UTILS_H
#include <memory>
#include "ascir.h"
namespace ascir {
namespace cg {
class CgContext {
 public:
  static CgContext *GetThreadLocalContext();
  static void SetThreadLocalContext(CgContext *context);

  const std::vector<Axis> &GetLoopAxes() const;
  const std::vector<AxisId> &GetLoopAxisIds() const;
  void SetLoopAxes(std::vector<Axis> axes);

 private:
  std::vector<Axis> loop_axes_;
  std::vector<AxisId> loop_axis_ids_cache_;  // 与 loop_axes_ 同源，避免反复创建
};

class LoopGuard {
 public:
  explicit LoopGuard(std::vector<Axis> axes);
  ~LoopGuard();

  template <typename... Args>
  static std::unique_ptr<LoopGuard> Create(Args... args) {
    std::vector<Axis> axes = {args...};
    return std::make_unique<LoopGuard>(std::move(axes));
  }

 private:
  CgContext context;
};

#define LOOP(...) for (auto guard = ascir::cg::LoopGuard::Create(__VA_ARGS__); guard != nullptr; guard = nullptr)

#define SET_SCHED_AXIS_IF_IN_CONTEXT()                 \
  do {                                                 \
    auto context = CgContext::GetThreadLocalContext(); \
    if (context != nullptr) {                          \
      op.attr.sched.axis = context->GetLoopAxisIds();  \
    }                                                  \
  } while (0)
}  // namespace cg
}  // namespace ascir
#endif  // AUTOFUSE_CG_UTILS_H
