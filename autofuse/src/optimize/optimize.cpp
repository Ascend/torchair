#include "optimize.h"

using namespace optimize;

Optimizer::Optimizer(const OptimizerOptions &options) {}

int Optimizer::Optimize(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &optimize_graphs) {
  return 0;
}

int Optimizer::GetApiInfo(const ascir::HintGraph &graph, ascir::ImplGraph &optimize_graph) {
  return 0;
}

int Optimizer::AutoScheduler(const ascir::HintGraph& graph, const ascir::ImplGraph optimizer_graph, std::vector<ascir::ImplGraph> &impl_graphs) {
  return 0;
}

int Optimizer::BufQueAlloc(const ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &optimize_graphs) {
  return 0;
}
