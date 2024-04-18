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
#define THROW(condition) if (!(condition)) throw std::runtime_error("Check Failed: " #condition);
struct LoopOption {
  bool pad_tensor_axes_to_loop;
};
class CgContext {
 public:
  static CgContext *GetThreadLocalContext();
  static std::shared_ptr<CgContext> GetSharedThreadLocalContext();
  static void SetThreadLocalContext(const std::shared_ptr<CgContext> &context);

  void SetOption(const LoopOption &option);
  const LoopOption &GetOption() const;

  const std::vector<Axis> &GetLoopAxes() const;
  const std::vector<AxisId> &GetLoopAxisIds() const;
  void SetLoopAxes(std::vector<Axis> axes);
  void PushLoopAxes(const std::vector<Axis> &axes);
  void PopBackLoopAxes(const std::vector<Axis> &axes);

  void SetBlockLoopEnd(AxisId id);
  AxisId GetBlockLoopEnd() const;

  void SetVectorizedLoopEnd(AxisId id);
  AxisId GetVectorizedLoopEnd() const;

  void SetLoopEnd(AxisId id);
  AxisId GetLoopEnd() const;

 private:
  LoopOption option_;
  std::vector<Axis> loop_axes_;
  std::vector<AxisId> loop_axis_ids_cache_;  // 与 loop_axes_ 同源，避免反复创建

  AxisId block_loop_end_{ID_NONE};
  AxisId vectorized_loop_end_{ID_NONE};
  AxisId loop_end_{ID_NONE};
};

class LoopGuard {
 public:
  explicit LoopGuard(std::vector<Axis> axes);
  ~LoopGuard();

  template <typename... Args>
  static std::unique_ptr<LoopGuard> Create(Args... args) {
    std::vector<Axis> axes = {args...};
    return Create(std::move(axes), {});
  }

  static std::unique_ptr<LoopGuard> Create(std::vector<Axis> axes, const LoopOption &option);

 private:
  std::vector<Axis> loop_axes_;
  std::shared_ptr<CgContext> context_;
};
using Axes = std::vector<Axis>;

class BlockLoopGuard {
  explicit BlockLoopGuard(std::vector<Axis> axes);
  ~BlockLoopGuard();
};

class VectorizedLoopGuard {
  explicit VectorizedLoopGuard(std::vector<Axis> axes);
  ~VectorizedLoopGuard();
};

#define INNER_LOOP_COUNTER_1(counter, ...)                                                               \
  for (auto guarder_##counter = ascir::cg::LoopGuard::Create(__VA_ARGS__); guarder_##counter != nullptr; \
       guarder_##counter = nullptr)
#define INNER_LOOP_COUNTER(counter, ...) INNER_LOOP_COUNTER_1(counter, __VA_ARGS__)
#define LOOP(...) INNER_LOOP_COUNTER(__COUNTER__, __VA_ARGS__)

#define OPTION_LOOP_COUNTER_1(counter, axes, option)                                                      \
  for (auto guarder_##counter = ascir::cg::LoopGuard::Create(axes, option); guarder_##counter != nullptr; \
       guarder_##counter = nullptr)
#define OPTION_LOOP_COUNTER(counter, axes, option) OPTION_LOOP_COUNTER_1(counter, axes, option)
#define OPTION_LOOP(axes, option) OPTION_LOOP_COUNTER(__COUNTER__, axes, option)

#define SET_SCHED_AXIS_IF_IN_CONTEXT()                            \
  do {                                                            \
    auto context = ascir::cg::CgContext::GetThreadLocalContext(); \
    if (context != nullptr) {                                     \
      op.attr.sched.axis = context->GetLoopAxisIds();             \
    }                                                             \
  } while (0)

void AddToGraphFollowOp(const ge::Operator &op, ge::Operator &new_op);
int64_t GenNextExecId(const ge::Operator &op);
int64_t GenNextExecId(const ascir::Graph &graph);

template <int OUTPUT_INDEX>
inline bool PadOutputViewToSched(ascir::OperatorOutput<OUTPUT_INDEX> &output) {
  auto context = ascir::cg::CgContext::GetThreadLocalContext();
  if (context == nullptr || !context->GetOption().pad_tensor_axes_to_loop) {
    return true;
  }

  // check if need pad
  auto &sched_ids = context->GetLoopAxisIds();
  auto origin_axis_ids = static_cast<std::vector<AxisId>>(output.axis);
  if (origin_axis_ids.size() == sched_ids.size()) {
    return origin_axis_ids == sched_ids;
  }

  // calc pad indexes, if op_i not iter to the end, means the axis order in tensor is different from sched
  // max: pad, positive: index of origin_axis_ids
  std::vector<size_t> indexes;
  size_t op_i = 0U;
  for (auto sched_axis_id : sched_ids) {
    if (op_i < origin_axis_ids.size() && sched_axis_id == origin_axis_ids.at(op_i)) {
      indexes.push_back(op_i++);
    } else {
      indexes.push_back(std::numeric_limits<size_t>::max());
    }
  }
  if (op_i != origin_axis_ids.size()) {
    return false;
  }

  // do pad
  auto origin_repeats = static_cast<std::vector<SizeExpr>>(output.repeats);
  auto origin_strides = static_cast<std::vector<SizeExpr>>(output.strides);
  std::vector<AxisId> padded_axis_ids;
  std::vector<SizeExpr> padded_repeats;
  std::vector<SizeExpr> padded_strides;
  for (size_t i = 0U; i < indexes.size(); ++i) {
    op_i = indexes[i];
    if (op_i == std::numeric_limits<size_t>::max()) {
      padded_axis_ids.push_back(sched_ids.at(i));
      padded_repeats.push_back(ascir::SizeExpr::One());
      padded_strides.push_back(ascir::SizeExpr::Zero());
    } else {
      padded_axis_ids.push_back(origin_axis_ids.at(op_i));
      padded_repeats.push_back(origin_repeats.at(op_i));
      padded_strides.push_back(origin_strides.at(op_i));
    }
  }

  output.axis = padded_axis_ids;
  output.repeats = padded_repeats;
  output.strides = padded_strides;

  return true;
}
}  // namespace cg
}  // namespace ascir
#endif  // AUTOFUSE_CG_UTILS_H
