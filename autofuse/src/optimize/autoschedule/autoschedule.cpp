#include "autoschedule.h"
#include "ascir.h"

namespace optimize::autoschedule {

void TilingGroup::SetReduceTilingGroup(std::vector<ascir::AxisId> yGroup, std::vector<ascir::AxisId> rGroup) {
  yGroup_ = std::move(yGroup);
  rGroup_ = std::move(rGroup);
}

bool TilingGroup::operator==(const TilingGroup& other) const {
  return xGroup_ == other.xGroup_ && yGroup_ == other.yGroup_ && rGroup_ == other.rGroup_;
}

std::size_t CalcHash(std::size_t init_hash, const std::vector<ascir::AxisId>& values) {
  for (auto& x : values) {
    init_hash ^= x + 0x9e3779b9 + (init_hash << 6) + (init_hash << 2);
  }
  return init_hash;
}

std::size_t TilingGroup::operator()(const TilingGroup& key) const {
  std::size_t hash = 0;
  hash = CalcHash(hash, xGroup_);
  hash = CalcHash(hash, yGroup_);
  hash = CalcHash(hash, rGroup_);
  return hash;
}

void TilingGroup::GenElewiseTilingGroup(const ascir::NodeView& node) {
  yGroup_ = node.attr.sched.axis;
}

void AutoSchedule::GenTilingCase() {
  for (auto& tiling : tilingGroup_) {
    tilingCase_.ubTilingIdY = tiling.first.YGroup()[1];
    tilingCase_.blockTilingId = tiling.first.YGroup()[0];
  }
}

void AutoSchedule::GenTilingGroup() {
  // 1.  遍历所有的节点，根据每个节点的轴生成tiling group
  // 2、 如果得到的tiling group没有被保存过，那么生成一个新的tiling group
  // 3. 相同tiling group中的节点对应相同的ting， size_align, stride_align
  // 4. data, load, store按照什么方式切分？ 分别生成当前存储中的group
  for (auto node : graph_.GetAllNodes()) {
    auto compute_type = GetComputeType(node);
    if (compute_type == ascir::COMPUTE_ELEWISE || ascir::COMPUTE_STORE || ascir::COMPUTE_LOAD) {
      TilingGroup tiling;
      tiling.GenElewiseTilingGroup(node);
      tilingGroup_[tiling].emplace_back(node);
    } else {
      throw std::runtime_error("Unsupported compute type");
    }
  }
}

void AutoSchedule::Tiling() {
  // split ub
  TileTiling(tilingCase_.ubTilingIdX, tilingCase_.ubTilingX);
  TileTiling(tilingCase_.ubTilingIdY, tilingCase_.ubTilingY);
  TileTiling(tilingCase_.ubTilingIdR, tilingCase_.ubTilingR);

  // split block
  if (tilingCase_.blockTilingId != -1) {
    tilingCase_.blockTiling = graph_.BlockSplit(tilingCase_.blockTilingId);
  }
}

void AutoSchedule::ApplyTiling() {
  for (auto node : graph_.GetAllNodes()) {
    ApplyTiling(node, tilingCase_.ubTilingIdX, tilingCase_.ubTilingX);
    ApplyTiling(node, tilingCase_.ubTilingIdY, tilingCase_.ubTilingY);
    ApplyTiling(node, tilingCase_.ubTilingIdR, tilingCase_.ubTilingR);
    ApplyTiling(node, tilingCase_.blockTilingId, tilingCase_.blockTiling);
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

void AutoSchedule::Vectorize() {
  for (const auto& tilingGroup: tilingGroup_) {
    vector<ascir::AxisId> vectorized_axis;
    FindInnerAxes(vectorized_axis, tilingGroup.first.XGroup(), tilingCase_.ubTilingIdX, tilingCase_.ubTilingX);
    FindInnerAxes(vectorized_axis, tilingGroup.first.YGroup(), tilingCase_.ubTilingIdY, tilingCase_.ubTilingY);
    FindInnerAxes(vectorized_axis, tilingGroup.first.RGroup(), tilingCase_.ubTilingIdR, tilingCase_.ubTilingR);

    for (auto node : tilingGroup.second) {
      if (node.attr.hint.compute_type != ascir::COMPUTE_DATA) {
        node.outputs[0].vectorized_axis = vectorized_axis;
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
void AutoSchedule::NodeNumber() {
  // 默认：0
  // broadcast前：1
  // reduce后broadcast前：2
  // gather和scatter的index：3
  for (auto node : graph_.GetAllNodes()) {
    if (GetComputeType(node) == ascir::COMPUTE_BROADCAST) {
      for (auto& output : node.outputs()) {
        ascir::SchAttr sch_attr{output.desc};
        sch_attr.group_id = 1;
      }
    } else if (GetComputeType(node) == ascir::COMPUTE_REDUCE) {
      for (auto& output : node.outputs()) {
        ascir::SchAttr sch_attr{output.desc};
        sch_attr.group_id = 1;
      }
    } else if (GetComputeType(node) != ascir::COMPUTE_STORE) {
      for (auto& output : node.outputs()) {
        ascir::SchAttr sch_attr{output.desc};
        sch_attr.group_id = 1;
      }
    }
    // todo gather and scatter
  }
}

void AutoSchedule::Scheduler() {
  // generator tiling group for every node
  GenTilingGroup();

  // enumerate the tiling split axes according to the tiling group
  GenTilingCase();

  // auto schedule
  Tiling();
  Vectorize();
  ApplyTiling();
}

}