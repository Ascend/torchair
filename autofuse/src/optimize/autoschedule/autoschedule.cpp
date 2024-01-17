#include "autoschedule.h"
#include <bits/stdint-intn.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <list>
#include <string>

#include "ascir.h"
#include "ascir_utils.h"

namespace optimize::autoschedule {
static constexpr int DEFAULT_GID = 0;
static constexpr int BEFOR_BRC_GID = 1;
static constexpr int AFTER_REDUCE_GID = 2;
static constexpr int DEFAULE_GROUP = -1;
static constexpr int QUEUE_SIZE = 4;

void PrintGroup(const std::string& name, const std::vector<ascir::AxisId>& group) {
  std::cout << name << ": [";
  for (auto& axis : group) {
    std::cout << axis << " ";
  }
  std::cout << "]\n";
}

void PrintAxesGroup(const std::string& node_name, const TilingGroup& axes_group) {
  std::cout << node_name << " TilingGroup: {\n";
  PrintGroup("  XGroup", axes_group.x_group_);
  PrintGroup("  YGroup", axes_group.y_group_);
  PrintGroup("  RGroup", axes_group.r_group_);
  std::cout << "  AxesOrder: [";
  for (auto& axis : axes_group.axes_order_) {
    std::cout << axis << " ";
  }
  std::cout << "]\n";
  std::cout << "}\n";
}

void PrintTilingCase(const TilingCase& tiling_case) {
  std::cout << "TilingCase: {\n";
  std::cout << "  ub_tiling_id_x: " << tiling_case.ub_tiling_id_x << "\n";
  std::cout << "  ub_tiling_id_y: " << tiling_case.ub_tiling_id_y << "\n";
  std::cout << "  ub_tiling_id_r: " << tiling_case.ub_tiling_id_r << "\n";
  std::cout << "  block_tiling_id: " << tiling_case.block_tiling_id << "\n";
  std::cout << "}\n";
}

void TilingGroup::SetReduceTilingGroup(std::vector<ascir::AxisId> y_group, std::vector<ascir::AxisId> r_group) {
  y_group_ = std::move(y_group);
  r_group_ = std::move(r_group);
}

bool TilingGroup::operator==(const TilingGroup& other) const {
  return x_group_ == other.x_group_ && y_group_ == other.y_group_ && r_group_ == other.r_group_;
}

std::size_t CalcHash(std::size_t init_hash, const std::vector<ascir::AxisId>& values) {
  for (auto& x : values) {
    init_hash ^= x + 0x9e3779b9 + (init_hash << 6) + (init_hash << 2);
  }
  return init_hash;
}

std::size_t TilingGroup::operator()(const TilingGroup& key) const {
  std::size_t hash = 0;
  hash = CalcHash(hash, x_group_);
  hash = CalcHash(hash, y_group_);
  hash = CalcHash(hash, r_group_);
  return hash;
}

void TilingGroup::GenElewiseTilingGroup(const ascir::NodeView& node) {
  y_group_ = node.attr.sched.axis;
  axes_order_.reserve(y_group_.size());
  for (int64_t i = 0; i < y_group_.size(); ++i) {
    axes_order_.push_back(i);
  }
}

std::vector<ascir::AxisId> CalcReduceAxes(const std::vector<ascir::SizeExpr>& src_strides,
                                          const std::vector<ascir::SizeExpr>& dst_strides,
                                          const std::vector<ascir::AxisId>& axes) {
  if (src_strides.size() != dst_strides.size()) {
    throw std::runtime_error("src strides length and dst strides length not equal");
  }
  if (src_strides.size() != axes.size()) {
    throw std::runtime_error("src strides length and axes length not equal");
  }
  std::vector<ascir::AxisId> reduce_axes;
  for (size_t i = 0; i < src_strides.size(); ++i) {
    if (src_strides[i] != dst_strides[i] && dst_strides[i] == 0) {
      reduce_axes.push_back(axes[i]);
    }
  }
  return reduce_axes;
}

void TilingGroup::GenReduceTilingGroup(ascir::NodeView& node) {
  const std::vector<ascir::AxisId>& axes = node.attr.sched.axis;
  axes_order_.resize(axes.size());
  r_group_ = CalcReduceAxes(node.inputs[0]->strides(), node.outputs[0].strides(), axes);
  int64_t y_order_index = 0;
  int64_t r_order_index = axes.size() - r_group_.size();
  for (size_t i = 0; i < axes.size(); ++i) {
    if (std::find(r_group_.begin(), r_group_.end(), axes[i]) == r_group_.end()) {
      y_group_.push_back(axes[i]);
      axes_order_[y_order_index++] = i;
    } else {
      axes_order_[r_order_index++] = i;
    }
  }
}

void TilingGroup::GenTransposeTilingGroup(const ascir::NodeView& node) {
  y_group_ = node.attr.sched.axis;
}

void TilingGroup::NormGroup() {
  if (x_group_.empty()) {
    x_group_.push_back(DEFAULE_GROUP);
  }
  if (r_group_.empty()) {
    r_group_.push_back(DEFAULE_GROUP);
  }
}

bool TilingGroup::IsEmpty() {
  return x_group_.empty() && y_group_.empty() && r_group_.empty();
}

void GetOuterAxes(const std::vector<ascir::AxisId>& axes_group,
                  const ascir::AxisId& ub_tiling_id,
                  const ascir::Axis& ub_tiling_outer_axis,
                  const std::vector<int64_t>& axes_order,
                  std::vector<ascir::AxisId>& outer_axes,
                  std::vector<int64_t>& outer_axes_index,
                  size_t axes_order_index) {
  for (size_t i = 0; i < axes_group.size(); i++) {
    if (axes_group[i] != ub_tiling_id) {
      outer_axes.push_back(axes_group[i]);
      outer_axes_index.push_back(axes_order[axes_order_index++]);
    } else if (axes_group[i] == ub_tiling_id) {
      outer_axes.push_back(ub_tiling_outer_axis.id);
      outer_axes_index.push_back(axes_order[axes_order_index++]);
      break;
    }
  }
}

void Scheduler::Tiling() {
  // split ub
  TileTiling(tiling_case_.ub_tiling_id_x, tiling_case_.ub_tiling_x);
  TileTiling(tiling_case_.ub_tiling_id_y, tiling_case_.ub_tiling_y);
  TileTiling(tiling_case_.ub_tiling_id_r, tiling_case_.ub_tiling_r);

  // reorder axes to original order, x group和y group都有值的时候才需要reorder
  std::vector<ascir::AxisId> non_reduce_outer_axes;
  std::vector<ascir::AxisId> reduce_outer_axes;
  std::vector<int64_t> non_reduce_outer_axes_index;
  std::vector<int64_t> reduce_outer_axes_index;
  size_t axes_order_index = 0;
  if (HasXGroup()) {
    GetOuterAxes(axes_group_.XGroup(), tiling_case_.ub_tiling_id_x, std::get<0>(tiling_case_.ub_tiling_x),
                 axes_group_.axes_order_, non_reduce_outer_axes, non_reduce_outer_axes_index, axes_order_index);
    axes_order_index += axes_group_.XGroup().size();
  }
  GetOuterAxes(axes_group_.YGroup(), tiling_case_.ub_tiling_id_y, std::get<0>(tiling_case_.ub_tiling_y),
               axes_group_.axes_order_, non_reduce_outer_axes, non_reduce_outer_axes_index, axes_order_index);
  axes_order_index += axes_group_.YGroup().size();
  // todo: transpose reorder, 考虑transpose分组中轴相同的场景

  if (HasRGroup()) {
    GetOuterAxes(axes_group_.RGroup(), tiling_case_.ub_tiling_id_r, std::get<0>(tiling_case_.ub_tiling_r),
                 axes_group_.axes_order_, reduce_outer_axes, reduce_outer_axes_index, axes_order_index);
  }
  // todo: transpose reorder

  // fuse outer non reduce axes and reduce axes
  if (non_reduce_outer_axes.size() > 1) {
    auto new_axis = graph_.MergeAxis(non_reduce_outer_axes);
    tiling_case_.block_tiling_id = new_axis.id;
  } else {
    if (non_reduce_outer_axes.size() != 1) {
      throw std::runtime_error("no block axis");
    }
    tiling_case_.block_tiling_id = non_reduce_outer_axes[0];
  }
  // 多核切分在reduce轴
  if (tiling_case_.reduce_is_block) {
    if (reduce_outer_axes.size() > 1) {
      reduce_outer_axes.insert(reduce_outer_axes.end(), non_reduce_outer_axes.begin(), non_reduce_outer_axes.end());
      auto new_axis = graph_.MergeAxis(reduce_outer_axes);
      tiling_case_.block_tiling_id = new_axis.id;
    } else {
      if (non_reduce_outer_axes.size() != 1) {
        throw std::runtime_error("no block axis");
      }
      tiling_case_.block_tiling_id = non_reduce_outer_axes[0];
    }
  }
  
  // split block
  if (tiling_case_.block_tiling_id != -1) {
    tiling_case_.block_tling = graph_.BlockSplit(tiling_case_.block_tiling_id);
    this->axes_order_.push_back(std::get<0>(tiling_case_.block_tling).id);
    this->axes_order_.push_back(std::get<1>(tiling_case_.block_tling).id);
    if (!tiling_case_.reduce_is_block && HasRGroup()) {
      this->axes_order_.insert(this->axes_order_.end(), reduce_outer_axes.begin(), reduce_outer_axes.end());
    }
  }
}

void Scheduler::ApplyTiling() {
  for (auto node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsGm(node)) {
      continue;
    }
    ApplyTiling(node, tiling_case_.ub_tiling_id_x, tiling_case_.ub_tiling_x);
    ApplyTiling(node, tiling_case_.ub_tiling_id_y, tiling_case_.ub_tiling_y);
    ApplyTiling(node, tiling_case_.ub_tiling_id_r, tiling_case_.ub_tiling_r);
    if (graph_.axis[tiling_case_.block_tiling_id].type == ascir::Axis::AXIS_TYPE_MERGED) {
      graph_.ApplyMerge(node, tiling_case_.block_tiling_id);
    }
    ApplyTiling(node, tiling_case_.block_tiling_id, tiling_case_.block_tling);
  }
}

void FindInnerAxes(vector<ascir::AxisId>& vectorize_axis,
                   const std::vector<ascir::AxisId>& axisGroup,
                   ascir::AxisId ubTilingId,
                   const std::tuple<ascir::Axis, ascir::Axis>& ubTiling) {
  bool findTileIn = false;
  for (const auto& axis : axisGroup) {
    if (axis == ubTilingId) {
      findTileIn = true;
      vectorize_axis.push_back(std::get<1>(ubTiling).id);
      continue;
    }

    if (findTileIn) {
      vectorize_axis.push_back(axis);
    }
  }
}

void Scheduler::Vectorize() {
  std::vector<ascir::AxisId> vectorized_axes;
  std::vector<int64_t> vectorized_axes_order;
  size_t last_ub_size = 0;
  size_t cum_size = 0;
  size_t begin_idx = 0;
  size_t end_idx = 0;
  if (this->HasXGroup()) {
    FindInnerAxes(vectorized_axes, axes_group_.XGroup(), tiling_case_.ub_tiling_id_x, tiling_case_.ub_tiling_x);
    last_ub_size = vectorized_axes.size();
    cum_size += axes_group_.XGroup().size();
    begin_idx = cum_size - last_ub_size;
    end_idx = begin_idx + vectorized_axes.size();
    vectorized_axes_order.insert(vectorized_axes_order.end(),
                                 this->axes_group_.axes_order_.begin() + begin_idx,
                                 this->axes_group_.axes_order_.begin() + end_idx);
  }
  
  FindInnerAxes(vectorized_axes, axes_group_.YGroup(), tiling_case_.ub_tiling_id_y, tiling_case_.ub_tiling_y);
  cum_size += axes_group_.YGroup().size();
  begin_idx = cum_size - (vectorized_axes.size() - last_ub_size);
  end_idx = begin_idx + (vectorized_axes.size() - last_ub_size);
  last_ub_size = vectorized_axes.size();
  vectorized_axes_order.insert(vectorized_axes_order.end(),
                               this->axes_group_.axes_order_.begin() + begin_idx,
                               this->axes_group_.axes_order_.begin() + end_idx);
  
  if (this->HasRGroup()) {
    cum_size += axes_group_.RGroup().size();
    FindInnerAxes(vectorized_axes, axes_group_.RGroup(), tiling_case_.ub_tiling_id_r, tiling_case_.ub_tiling_r);
    begin_idx = cum_size - (vectorized_axes.size() - last_ub_size);
    end_idx = begin_idx + (vectorized_axes.size() - last_ub_size);
    vectorized_axes_order.insert(vectorized_axes_order.end(),
                                 this->axes_group_.axes_order_.begin() + begin_idx,
                                 this->axes_group_.axes_order_.begin() + end_idx);
  }
  
  // reorder vectorized axis
  vector<int64_t> base_order(vectorized_axes_order.size(), 0);
  for (size_t i = 0; i < base_order.size(); i++) {
    base_order[i] = i;
  }
  std::sort(base_order.begin(), base_order.end(), [&vectorized_axes_order](int64_t a, int64_t b) {
    return vectorized_axes_order[a] < vectorized_axes_order[b];
  });
  for (size_t i = 0; i < base_order.size(); i++) {
    this->axes_order_.push_back(vectorized_axes[i]);
  }

  for (auto node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsGm(node)) {
      continue;
    }
    node.outputs[0].vectorized_axis = vectorized_axes;
  }
}

void Scheduler::Reorder() {
  for (auto node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsGm(node)) {
      continue;
    }
    graph_.ApplyReorder(node, this->axes_order_);
  }
}

void Scheduler::UpdateUpGroupId(ascir::NodeView node, int group_id) {
  for (auto& input : node.inputs()) {
    ascir::SchAttr sch_attr{input->desc};
    sch_attr.group_id = group_id;
    UpdateUpGroupId(ascir::NodeView(input->Owner()), group_id);
  }
}

void Scheduler::UpdateDownGroupId(ascir::NodeView node, int group_id) {
  for (auto& output : node.outputs()) {
    ascir::SchAttr sch_attr{output.desc};
    sch_attr.group_id = group_id;
    UpdateDownGroupId(ascir::NodeView(output->GetOwnerNode()), group_id);
  }
}

void SetGroupId(ascir::NodeView node, int group_id) {
  for (auto& output : node.outputs()) {
    ascir::SchAttr sch_attr{output.desc};
    sch_attr.group_id = group_id;
  }
}

void HasAllMark(ascir::NodeView node, bool& has_mark_group_id) {
  for (auto& output : node.outputs()) {
    ascir::SchAttr sch_attr{output.desc};
    if (sch_attr.group_id == -1) {
      has_mark_group_id = false;
      break;
    }
  }
}

void Scheduler::AssignGroupId() {
  std::array<std::list<int>, QUEUE_SIZE> busy;
  std::array<std::list<int>, QUEUE_SIZE> free;
  for (auto node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsStore(ScheduleUtils::GetComputeType(node)) ||
        ScheduleUtils::IsData(ScheduleUtils::GetComputeType(node))) {
      continue;
    }
    for (auto& output : node.outputs()) {
      auto sch_attr  = ascir::SchAttr(output.desc);
      sch_attr.depends = output->GetPeerInDataNodesSize();
      int64_t reuse_id = 0;
      if (!free[sch_attr.group_id].empty()) {
        reuse_id = free[sch_attr.group_id].front();
        free[sch_attr.group_id].pop_front();
      } else {
        reuse_id = busy[sch_attr.group_id].size();
      }
      busy[sch_attr.group_id].push_back(reuse_id);
      output.opt.reuse_id = reuse_id;
    }

    for (auto& input: node.inputs()) {
      auto sch_attr = ascir::SchAttr(input->desc);
      sch_attr.depends = sch_attr.depends - 1;
      if (sch_attr.depends == 0) {
        busy[sch_attr.group_id].remove_if([&](int64_t n) {
          return n == input->opt.reuse_id;
        });
        free[sch_attr.group_id].push_back(input->opt.reuse_id);
      }
    }
  }
}

// broadcast前的节点分为一组
// reduce后braodcast前的节点分为一组
// gather和scatter的index分为一组
// 其他节点分为一组
// 定义一个数据结构，重新遍历所有节点，根据节点的size，将节点分为不同的group
// 从输入开始遍历图，找到节点所属的组，然后从那个组中获取编号，每个组维护一个队列。
// 按照节点顺序遍历计算图，通过GetAllNodes得到的即为此顺序。
void Scheduler::NodeNumber() {
  // 默认：0
  // broadcast前：1
  // reduce后broadcast前：2
  // gather和scatter的index：3
  for (auto node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsStore(node)) {
      SetGroupId(node, -1);
    }
  }

  for (auto node : graph_.GetAllNodes()) {
    bool has_mark_group_id = true;
    HasAllMark(node, has_mark_group_id);
    if (has_mark_group_id) {
      continue;
    }

    if (ScheduleUtils::IsBroadcast(node)) {
      SetGroupId(node, DEFAULT_GID);
      UpdateUpGroupId(node, BEFOR_BRC_GID);
    } else if (ScheduleUtils::IsReduce(node)) {
      SetGroupId(node, AFTER_REDUCE_GID);
      UpdateDownGroupId(node, AFTER_REDUCE_GID);
    } else if (!ScheduleUtils::IsStore(node) && !ScheduleUtils::IsData(node) && !ScheduleUtils::IsWorkspace(node)) {
      SetGroupId(node, DEFAULT_GID);
    }
    // todo gather and scatter
  }

  AssignGroupId();
}

void Scheduler::DoScheduler() {
  Tiling();
  Vectorize();
  ApplyTiling();
  Reorder();
  NodeNumber();
}

void AutoSchedule::SingelNodeAxesGroup(ascir::NodeView& node, TilingGroup& axes_group) {
  if (ScheduleUtils::IsElewise(node) ||
      ScheduleUtils::IsBroadcast(node) ||
      ScheduleUtils::IsLoad(node) ||
      ScheduleUtils::IsStore(node)) {
    axes_group.GenElewiseTilingGroup(node);
  } else if (ScheduleUtils::IsReduce(node)) {
    axes_group.GenReduceTilingGroup(node);
  } else if (ScheduleUtils::IsTranspose(node)) {
    axes_group.GenTransposeTilingGroup(node);
  } else {
    throw std::runtime_error("Unsupported compute type");
  }
}

bool IsYGroup(const TilingGroup& single_node_axes_group) {
  return single_node_axes_group.XGroup().empty() && single_node_axes_group.RGroup().empty();
}

bool IsYRGroup(const TilingGroup& single_node_axes_group) {
  return single_node_axes_group.XGroup().empty() && !single_node_axes_group.RGroup().empty();
}

bool IsXYGroup(const TilingGroup& single_node_axes_group) {
  return !single_node_axes_group.XGroup().empty() && single_node_axes_group.RGroup().empty();
}

bool IsXYRGroup(const TilingGroup& single_node_axes_group) {
  return !single_node_axes_group.XGroup().empty() && !single_node_axes_group.RGroup().empty();
}

bool CheckYAndYRGroup(const std::vector<ascir::AxisId>& cur_y_group,
                      const std::vector<ascir::AxisId>& new_y_group,
                      const std::vector<ascir::AxisId>& new_r_group,
                      const std::vector<int64_t>& new_axes_order) {
  if (cur_y_group.size() != (new_y_group.size() + new_r_group.size())) {
    return false;
  }
  for (size_t i = 0; i < new_y_group.size(); i++) {
    if (cur_y_group[new_axes_order[i]] != new_y_group[i]) {
      return false;
    }
  }
  int64_t y_size = new_y_group.size();
  for (size_t i = 0; i < new_r_group.size(); i++) {
    if (cur_y_group[new_axes_order[y_size + i]] != new_r_group[i]) {
      return false;
    }
  }
  return true;
}

// 在此之前不应该做任何的合轴，否则很难推断出轴关系
// 1. (1, y0, 1) merge (1, y1, 1) ==> (1, max(y0, y1), 1), 如果节点2是broadcast节点，要求节点1的tiling group与节点2的输入节点的tiling group一致，否则应该满足y0 == y1
// 2. (1, y0, 1) merge (1, y1, r1) ==> (1, y1, r1), 要求y0 == y1 U r1
// 3. (1, y0, 1) merge (x1, y1, 1) ==> (x1, y1, 1)
// 4. (1, y0, 1) merge (x1, y1, r1) ==> (x1, y1, r1)
// 5. (1, y0, r0) merge (1, y1, 1) ==> (1, y0, r0), 要求y1 == y0 U r0
// 6. (1, y0, r0) merge (1, y1, r1) ==> (1, y0, r0), 要求y1 == y0, r1 == r0
// 7. (1, y0, r0) merge (x1, y1, 1) ==> (x2, y2, r0)
// 8. (1, y0, r0) merge (x1, y1, r1) ==> (x1, y1, r1)
// 9. (x0, y0, 1) merge (1, y1, 1) ==> (x0, y0, 1)
// 10. (x0, y0, 1) merge (1, y1, r1) ==> (x2, y2, r1)
// 11. (x0, y0, 1) merge (x1, y1, 1) ==> (x0, y0, 1)
// 12. (x0, y0, 1) merge (x1, y1, r1) ==> (x1, y1, r1)
// 13. (x0, y0, r0) merge (1, y1, 1) ==> (x0, y0, r0)
// 14. (x0, y0, r0) merge (1, y1, r1) ==> (x0, y0, r0)
// 15. (x0, y0, r0) merge (x1, y1, 1) ==> (x0, y0, r0)
// 16. (x0, y0, r0) merge (x1, y1, r1) ==> (x0, y0, r0)
bool AutoSchedule::MergeAxesGroup(ascir::NodeView& node, const TilingGroup& node_axes_group) {
  if (this->axes_group_.IsEmpty()) {
    this->axes_group_ = node_axes_group;
    return true;
  }
  if (this->axes_group_ == node_axes_group) {
    return true;
  } else if (IsYGroup(this->axes_group_) && IsYGroup(node_axes_group)) {
    // y0 != y1仅有一个场景能支持融合，从elementwise到brodcast的过度
    if (this->axes_group_.YGroup().size() != node_axes_group.YGroup().size()) {
      return false;
    }
    for (size_t i = 0; i < this->axes_group_.YGroup().size(); i++) {
      if (this->axes_group_.YGroup()[i] == DEFAULE_GROUP) {
        this->axes_group_.y_group_[i] = node_axes_group.YGroup()[i];
      } else if (node_axes_group.YGroup()[i] != DEFAULE_GROUP &&
                 this->axes_group_.YGroup()[i] != node_axes_group.YGroup()[i]) {
        return false;
      }
    }
  } else if (IsYGroup(this->axes_group_) && IsYRGroup(node_axes_group)) {
    if (!CheckYAndYRGroup(this->axes_group_.YGroup(), node_axes_group.YGroup(),
                          node_axes_group.RGroup(), node_axes_group.axes_order_)) {
      return false;
    }
    this->axes_group_ = node_axes_group;
  } else if (IsYRGroup(this->axes_group_) && IsYGroup(node_axes_group)) {
    return CheckYAndYRGroup(node_axes_group.YGroup(), this->axes_group_.YGroup(),
                            this->axes_group_.RGroup(), this->axes_group_.axes_order_);
  } else if (IsYRGroup(this->axes_group_) && IsYRGroup(node_axes_group)) {
    // 此种情况要求y0 == y1, r0 == r1，所以走到此分支代表融合失败
    return false;
  } else {
    return false;
  }
  return true;
}

/**
 * @brief 算子级别的axes group生成
 * 1. 对于每个节点，根据其compute type生成不同的axes group
 * 2. 将生成的axes group与当前的axes group进行合并生成一个新的axes group
 */
void AutoSchedule::GenAxesGroup() {
  for (auto node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsGm(node)) {
      continue;
    }
    TilingGroup single_node_axes_group;
    SingelNodeAxesGroup(node, single_node_axes_group);
    if (!MergeAxesGroup(node, single_node_axes_group)) {
      throw std::runtime_error("Axes group merge fail, cannot fusion");
    }
  }
  axes_group_.NormGroup();
}

void AutoSchedule::GenTilingCase(std::vector<TilingCase>& tiling_cases) {
  // 生成通用pattern
  // 遍历所有的group，分别从每个group中取出1个值，组成所有的tiling case
  for (size_t i = 0; i < axes_group_.XGroup().size(); i++) {
    for (size_t j = 0; j < axes_group_.YGroup().size(); j++) {
      for (size_t k = 0; k < axes_group_.RGroup().size(); k++) {
        TilingCase tiling_case;
        tiling_case.schedule_pattern = SCHEDULE_GENERAL;
        if (axes_group_.XGroup()[i] != DEFAULE_GROUP) {
          tiling_case.ub_tiling_id_x = axes_group_.XGroup()[i];
        }
        tiling_case.ub_tiling_id_y = axes_group_.YGroup()[j];
        if (axes_group_.RGroup()[k] != DEFAULE_GROUP) {
          tiling_case.ub_tiling_id_r = axes_group_.RGroup()[k];
        }
        tiling_case.block_tiling_id = 0;
        tiling_cases.push_back(tiling_case);
        // todo: 只有输出是reduce，并且中间没有reduce的情况才能block切分在多核轴
        // 或者在没有norm结构的reduce上可以切分reduce轴，此时需要生成软同步
        bool is_reduce_output = false;
        if (is_reduce_output && axes_group_.RGroup()[i] != DEFAULE_GROUP) {
          TilingCase block_tiling_case = tiling_case;
          block_tiling_case.block_tiling_id = 1;
          block_tiling_case.reduce_is_block = true;
          tiling_cases.push_back(block_tiling_case);
        }
      }
    }
  }

  // 生成对齐pattern
  // todo

  // 生成UB内对齐pattern
  // todo
}

std::string GetTilingCaseStr(std::string graph_name, const TilingCase& tiling_case) {
  std::stringstream ss;
  ss << graph_name;
  if (tiling_case.schedule_pattern == SCHEDULE_GENERAL) {
    ss << "_general";
  } else if (tiling_case.schedule_pattern == SCHEDULE_ALIGN) {
    ss << "_align";
  } else if (tiling_case.schedule_pattern == SCHEDULE_INNER_ALIGN) {
    ss << "_inner_align";
  }

  auto IdStr = ascir::utils::IdentifierToStr;
  ss << "_" << IdStr(tiling_case.block_tiling_id);
  ss << "_" << IdStr(tiling_case.ub_tiling_id_x) << "_" << IdStr(tiling_case.ub_tiling_id_y) << "_" << IdStr(tiling_case.ub_tiling_id_r);
  return ss.str();
}

void AutoSchedule::DoAutoSchedule() {
  this->GenAxesGroup();

  std::vector<TilingCase> tiling_cases;
  this->GenTilingCase(tiling_cases);
  for (auto& tiling_case : tiling_cases) {
    ascir::Graph optimize_graph(GetTilingCaseStr(graph_.GetName(), tiling_case).c_str());
    optimize_graph.CopyFrom(graph_);
    optimize_graph.SortByExecOrder();
    Scheduler scheduler(optimize_graph, this->axes_group_, tiling_case);
    scheduler.DoScheduler();
    this->sch_graphs_.emplace_back(optimize_graph);
  }
}

}
