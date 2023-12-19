#ifndef __OPTIMIZE_AUTOSCHEDULE_AUTOSCHEDULE_H__
#define __OPTIMIZE_AUTOSCHEDULE_AUTOSCHEDULE_H__

#include "ascir.h"
#include <tuple>
#include <unordered_map>

namespace optimize::autoschedule {
class TilingGroup {
public:
  TilingGroup() = default;
  explicit TilingGroup(std::vector<ascir::AxisId> xGroup, std::vector<ascir::AxisId> yGroup, std::vector<ascir::AxisId> rGroup)
   : xGroup_(std::move(xGroup)), yGroup_(std::move(yGroup)), rGroup_(std::move(rGroup)) {};
  explicit TilingGroup(std::vector<ascir::AxisId> xGroup, std::vector<ascir::AxisId> yGroup)
   : xGroup_(std::move(xGroup)), yGroup_(std::move(yGroup)) {};
  explicit TilingGroup(std::vector<ascir::AxisId> yGroup) : yGroup_(std::move(yGroup)) {};
  void SetReduceTilingGroup(std::vector<ascir::AxisId> yGroup, std::vector<ascir::AxisId> rGroup);

  bool operator==(const TilingGroup& other) const;
  std::size_t operator()(const TilingGroup& key) const;

  inline const std::vector<ascir::AxisId>& XGroup() const {
    return xGroup_;
  }

  inline const std::vector<ascir::AxisId>& YGroup() const {
    return yGroup_;
  }

  inline const std::vector<ascir::AxisId>& RGroup() const {
    return rGroup_;
  }

  void GenElewiseTilingGroup(const ascir::NodeView& node);

private:
  std::vector<ascir::AxisId> xGroup_;
  std::vector<ascir::AxisId> yGroup_;
  std::vector<ascir::AxisId> rGroup_;
};

struct TilingCase {
  ascir::AxisId ubTilingIdX = -1;
  ascir::AxisId ubTilingIdY = -1;
  ascir::AxisId ubTilingIdR = -1;
  ascir::AxisId blockTilingId = -1;
  std::tuple<ascir::Axis, ascir::Axis> ubTilingX;
  std::tuple<ascir::Axis, ascir::Axis> ubTilingY;
  std::tuple<ascir::Axis, ascir::Axis> ubTilingR;
  std::tuple<ascir::Axis, ascir::Axis> blockTiling;
};

struct EqualTilingGroup {
  bool operator()(const TilingGroup& lhs, const TilingGroup& rhs) const {
    return lhs.XGroup() == rhs.XGroup() && lhs.YGroup() == rhs.YGroup() && lhs.RGroup() == rhs.RGroup();
  }
};

class AutoSchedule {
public:
  AutoSchedule() = delete;
  explicit AutoSchedule(ascir::Graph& graph) : graph_(graph) {};

  inline ascir::ComputeType GetComputeType(ascir::NodeView& nodeView) {
    return nodeView.attr.hint.compute_type;
  }

  void GenTilingCase();
  void GenTilingGroup();
  void Scheduler();
  void Tiling();
  void ApplyTiling();
  void Vectorize();
  void NodeNumber();

private:
  inline void TileTiling(ascir::AxisId tileId, std::tuple<ascir::Axis, ascir::Axis>& tiledAxes) {
    if (tileId != -1) {
      tiledAxes = graph_.TileSplit(tileId);
    }
  }

  inline void ApplyTiling(ascir::NodeView& node, ascir::AxisId tileId,
                          const std::tuple<ascir::Axis, ascir::Axis>& tiledAxes) {
    if (tileId != -1) {
      graph_.ApplySplit(node, std::get<0>(tiledAxes).id,
                        std::get<1>(tiledAxes).id, tileId);
    }
  }

private:
  ascir::Graph& graph_;
  std::unordered_map<TilingGroup, std::vector<ascir::NodeView>, TilingGroup, EqualTilingGroup> tilingGroup_;
  struct TilingCase tilingCase_;
};
}

#endif