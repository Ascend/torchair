#ifndef __OPTIMIZE_AUTOSCHEDULE_AUTOSCHEDULE_H__
#define __OPTIMIZE_AUTOSCHEDULE_AUTOSCHEDULE_H__

#include "ascir.h"
#include <tuple>
#include <unordered_map>

namespace optimize::autoschedule {

enum SchedulePattern {
  SCHEDULE_GENERAL,
  SCHEDULE_ALIGN,
  SCHEDULE_INNER_ALIGN
};

class ScheduleUtils {
public:
  static inline ascir::ComputeType GetComputeType(ascir::NodeView& node_view) {
    return node_view.attr.hint.compute_type;
  }

  static inline bool IsElewise(ascir::ComputeType compute_Type) {
    return compute_Type == ascir::COMPUTE_ELEWISE;
  }

  static inline bool IsElewise(ascir::NodeView& node_view) {
    return node_view.attr.hint.compute_type == ascir::COMPUTE_ELEWISE;
  }

  static inline bool IsBroadcast(ascir::ComputeType compute_Type) {
    return compute_Type == ascir::COMPUTE_BROADCAST;
  }

  static inline bool IsBroadcast(ascir::NodeView& node_view) {
    return node_view.attr.hint.compute_type == ascir::COMPUTE_BROADCAST;
  }

  static inline bool IsReduce(ascir::ComputeType compute_Type) {
    return compute_Type == ascir::COMPUTE_REDUCE;
  }

  static inline bool IsReduce(ascir::NodeView& node_view) {
    return node_view.attr.hint.compute_type == ascir::COMPUTE_REDUCE;
  }

  static inline bool IsTranspose(ascir::ComputeType compute_Type) {
    return compute_Type == ascir::COMPUTE_TRANPOSE;
  }

  static inline bool IsTranspose(ascir::NodeView& node_view) {
    return node_view.attr.hint.compute_type == ascir::COMPUTE_TRANPOSE;
  }

  static inline bool IsData(ascir::ComputeType compute_Type) {
    return compute_Type == ascir::COMPUTE_DATA;
  }

  static inline bool IsData(ascir::NodeView& node_view) {
    return node_view.attr.hint.compute_type == ascir::COMPUTE_DATA;
  }

  static inline bool IsLoad(ascir::ComputeType compute_Type) {
    return compute_Type == ascir::COMPUTE_LOAD;
  }

  static inline bool IsLoad(ascir::NodeView& node_view) {
    return node_view.attr.hint.compute_type == ascir::COMPUTE_LOAD;
  }

  static inline bool IsStore(ascir::ComputeType compute_Type) {
    return compute_Type == ascir::COMPUTE_STORE;
  }

  static inline bool IsStore(ascir::NodeView& node_view) {
    return node_view.attr.hint.compute_type == ascir::COMPUTE_STORE;
  }
};

class TilingGroup {
public:
  TilingGroup() = default;
  explicit TilingGroup(std::vector<ascir::AxisId> x_group, std::vector<ascir::AxisId> y_group, std::vector<ascir::AxisId> r_group)
   : x_group_(std::move(x_group)), y_group_(std::move(y_group)), r_group_(std::move(r_group)) {};
  explicit TilingGroup(std::vector<ascir::AxisId> x_group, std::vector<ascir::AxisId> y_group)
   : x_group_(std::move(x_group)), y_group_(std::move(y_group)) {};
  explicit TilingGroup(std::vector<ascir::AxisId> y_group) : y_group_(std::move(y_group)) {};
  void SetReduceTilingGroup(std::vector<ascir::AxisId> y_group, std::vector<ascir::AxisId> r_group);

  bool operator==(const TilingGroup& other) const;
  std::size_t operator()(const TilingGroup& key) const;

  inline const std::vector<ascir::AxisId>& XGroup() const {
    return x_group_;
  }

  inline const std::vector<ascir::AxisId>& YGroup() const {
    return y_group_;
  }

  inline const std::vector<ascir::AxisId>& RGroup() const {
    return r_group_;
  }

  inline const std::vector<ascir::AxisId>& BlockTilingGroup() const {
    return r_group_;
  }

  void GenElewiseTilingGroup(const ascir::NodeView& node);
  void GenReduceTilingGroup(ascir::NodeView& node);
  void GenTransposeTilingGroup(const ascir::NodeView& node);
  void NormGroup();
  bool IsEmpty();

private:
  std::vector<ascir::AxisId> x_group_;
  std::vector<ascir::AxisId> y_group_;
  std::vector<ascir::AxisId> r_group_;
};

typedef struct TilingCase {
  ascir::AxisId ub_tiling_id_x = -1;
  ascir::AxisId ub_tiling_id_y = -1;
  ascir::AxisId ub_tiling_id_r = -1;
  ascir::AxisId block_tiling_id = -1;
  std::tuple<ascir::Axis, ascir::Axis> ub_tiling_x;
  std::tuple<ascir::Axis, ascir::Axis> ub_tiling_y;
  std::tuple<ascir::Axis, ascir::Axis> ub_tiling_r;
  std::tuple<ascir::Axis, ascir::Axis> block_tling;
  SchedulePattern schedule_pattern = SCHEDULE_GENERAL;
} TilingCase;

struct EqualTilingGroup {
  bool operator()(const TilingGroup& lhs, const TilingGroup& rhs) const {
    return lhs.XGroup() == rhs.XGroup() && lhs.YGroup() == rhs.YGroup() && lhs.RGroup() == rhs.RGroup();
  }
};

class Scheduler {
public:
  Scheduler() = delete;
  explicit Scheduler(ascir::ImplGraph& graph, const TilingGroup& axes_group_, TilingCase tiling_case)
   : graph_(graph), axes_group_(axes_group_), tiling_case_(std::move(tiling_case)) {};

  void DoScheduler();
  void Tiling();
  void ApplyTiling();
  void Vectorize();
  void NodeNumber();

private:
  void UpdateUpGroupId(ascir::NodeView node, int group_id);
  void UpdateDownGroupId(ascir::NodeView node, int group_id);
  void AssignGroupId();
  inline void TileTiling(ascir::AxisId tile_id, std::tuple<ascir::Axis, ascir::Axis>& tiled_axes) {
    if (tile_id != -1) {
      tiled_axes = graph_.TileSplit(tile_id);
      if (tiling_case_.block_tiling_id == tile_id) {
        tiling_case_.block_tiling_id = std::get<0>(tiled_axes).id;
      }
    }
  }

  inline void ApplyTiling(ascir::NodeView& node, ascir::AxisId tile_id,
                          const std::tuple<ascir::Axis, ascir::Axis>& tiled_axes) {
    if (tile_id != -1) {
      graph_.ApplySplit(node, std::get<0>(tiled_axes).id,
                        std::get<1>(tiled_axes).id, tile_id);
    }
  }

private:
  ascir::ImplGraph& graph_;
  const TilingGroup& axes_group_;
  TilingCase tiling_case_;
};

class AutoSchedule {
public:
  AutoSchedule() = delete;
  explicit AutoSchedule(const ascir::ImplGraph& graph, std::vector<ascir::ImplGraph>& sch_graphs)
   : graph_(graph), sch_graphs_(sch_graphs) {};

  void DoAutoSchedule();
  void GenAxesGroup();
  void GenTilingCase(std::vector<TilingCase>& tiling_cases);
  static void SingelNodeAxesGroup(ascir::NodeView& node, TilingGroup& axes_group);

private:
  void MergeAxesGroup(const TilingGroup& single_node_axes_group);

private:
  const ascir::ImplGraph& graph_;
  std::vector<ascir::ImplGraph>& sch_graphs_;
  TilingGroup axes_group_;
};
}

#endif