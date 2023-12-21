#ifndef __AUTOFUSE_OPTIMIZE_H__
#define __AUTOFUSE_OPTIMIZE_H__

#include "ascir.h"

namespace optimize {
struct OptimizerOptions {};

class Optimizer {
 public:
  explicit Optimizer(const OptimizerOptions& options);

  int Optimize(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &optimize_graphs);

  /**
   * 根据HintGraph，设置ImplGraph中的Api信息
   * @param [in] graph 原始图
   * @param [in,out] optimize_graph 优化后的图，同时也将api信息设置在这个图上
   */
  int GetApiInfo(const ascir::HintGraph& graph, ascir::ImplGraph &optimize_graph);

  /**
   * 自动调度
   * @param [in] graph 原始图
   * @param [in] optimizer_graph 优化后的图
   * @param [out] impl_graphs 输出不同tiling策略切分的图
   */
  int AutoScheduler(const ascir::HintGraph& graph, const ascir::ImplGraph& optimizer_graph, std::vector<ascir::ImplGraph> &impl_graphs);

  /**
   * Buf/Que 分配
   * @param [in] graph 原始图
   * @param [in,out] impl_graphs schduler后的图，同时也将内存分配设置到这些图上
   */
  int BufQueAlloc(const ascir::HintGraph& graph, std::vector<ascir::ImplGraph> &impl_graphs);
};
} // namespace optimize

#endif
