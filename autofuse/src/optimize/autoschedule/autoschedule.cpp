#include "autoschedule.h"
#include <array>
#include <cmath>
#include <cstddef>
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
static constexpr int DEFAULE_GROUP = 1;
static constexpr int QUEUE_SIZE = 4;

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
    if (src_strides[i] != dst_strides[i]) {
      reduce_axes.push_back(axes[i]);
    }
  }
  return reduce_axes;
}

void TilingGroup::GenReduceTilingGroup(ascir::NodeView& node) {
  const std::vector<ascir::AxisId>& axes = node.attr.sched.axis;
  r_group_ = CalcReduceAxes(node.inputs[0]->strides(), node.outputs[0].strides(), axes);
  for (size_t i = 0; i < axes.size(); ++i) {
    if (std::find(r_group_.begin(), r_group_.end(), axes[i]) == r_group_.end()) {
      y_group_.push_back(axes[i]);
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

void Scheduler::Tiling() {
  // split ub
  TileTiling(tiling_case_.ub_tiling_id_x, tiling_case_.ub_tiling_x);
  TileTiling(tiling_case_.ub_tiling_id_y, tiling_case_.ub_tiling_y);
  TileTiling(tiling_case_.ub_tiling_id_r, tiling_case_.ub_tiling_r);

  // split block
  if (tiling_case_.block_tiling_id != -1) {
    tiling_case_.block_tling = graph_.BlockSplit(tiling_case_.block_tiling_id);
  }
}

void Scheduler::ApplyTiling() {
  for (auto node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsData(ScheduleUtils::GetComputeType(node))) {
      continue;
    }
    ApplyTiling(node, tiling_case_.ub_tiling_id_x, tiling_case_.ub_tiling_x);
    ApplyTiling(node, tiling_case_.ub_tiling_id_y, tiling_case_.ub_tiling_y);
    ApplyTiling(node, tiling_case_.ub_tiling_id_r, tiling_case_.ub_tiling_r);
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
  vector<ascir::AxisId> vectorized_axis;
  FindInnerAxes(vectorized_axis, axes_group_.XGroup(), tiling_case_.ub_tiling_id_x, tiling_case_.ub_tiling_x);
  FindInnerAxes(vectorized_axis, axes_group_.YGroup(), tiling_case_.ub_tiling_id_y, tiling_case_.ub_tiling_y);
  FindInnerAxes(vectorized_axis, axes_group_.RGroup(), tiling_case_.ub_tiling_id_r, tiling_case_.ub_tiling_r);

  for (auto node : graph_.GetAllNodes()) {
    if (node.attr.hint.compute_type != ascir::COMPUTE_DATA) {
      node.outputs[0].vectorized_axis = vectorized_axis;
    }
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
    } else if (!ScheduleUtils::IsStore(node) && !ScheduleUtils::IsData(node)) {
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

void AutoSchedule::MergeAxesGroup(const TilingGroup& single_node_axes_group) {
  if (this->axes_group_.IsEmpty()) {
    this->axes_group_ = single_node_axes_group;
    return;
  }
  if (this->axes_group_ == single_node_axes_group) {
    return;
  } else {
     throw std::runtime_error("Unsupported axes group");
  }
}

/**
 * @brief 算子级别的axes group生成
 * 1. 对于每个节点，根据其compute type生成不同的axes group
 * 2. 将生成的axes group与当前的axes group进行合并生成一个新的axes group
 */
void AutoSchedule::GenAxesGroup() {
  for (auto node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsData(node)) {
      continue;
    }
    TilingGroup single_node_axes_group;
    SingelNodeAxesGroup(node, single_node_axes_group);
    MergeAxesGroup(single_node_axes_group);
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
          tiling_case.ub_tiling_id_x = i;
        }
        tiling_case.ub_tiling_id_y = j;
        if (axes_group_.RGroup()[i] != DEFAULE_GROUP) {
          tiling_case.ub_tiling_id_r = k;
        }
        tiling_case.block_tiling_id = 0;
        tiling_cases.push_back(tiling_case);
        if (axes_group_.RGroup().size() > 1 ||
            (axes_group_.RGroup().size() == 1 && axes_group_.RGroup()[0] != DEFAULE_GROUP)) {
          TilingCase block_tiling_case = tiling_case;
          block_tiling_case.block_tiling_id = 1;
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
    Scheduler scheduler(optimize_graph, this->axes_group_, tiling_case);
    scheduler.DoScheduler();
    this->sch_graphs_.emplace_back(optimize_graph);
  }
}

}
