#include "optimize.h"

#include "autoschedule/autoschedule.h"

using namespace optimize;

Optimizer::Optimizer(const OptimizerOptions &options) {}

int Optimizer::Optimize(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &optimize_graphs) {
  ascir::ImplGraph optimize_graph((graph.GetName() + "_optmize").c_str());
  optimize_graph.CopyFrom(graph);

  int ret = this->GetApiInfo(graph, optimize_graph);
  if (ret != 0) {
    return ret;
  }

  ret = this->AutoScheduler(graph, optimize_graph, optimize_graphs);
  if (ret != 0) {
    return ret;
  }

  ret = this->BufQueAlloc(graph, optimize_graphs);
  if (ret != 0) {
    return ret;
  }
  return 0;
}

int Optimizer::GetApiInfo(const ascir::HintGraph &graph, ascir::ImplGraph &optimize_graph) {
  return 0;
}

int Optimizer::AutoScheduler(const ascir::HintGraph& graph, const ascir::ImplGraph& optimizer_graph, std::vector<ascir::ImplGraph> &impl_graphs) {
  auto& impl1 = impl_graphs.emplace_back(ascir::ImplGraph((optimizer_graph.GetName() + "_tiling1").c_str()));
  impl1.CopyFrom(optimizer_graph);
  auto scheduler = autoschedule::AutoSchedule(impl1);
  scheduler.Scheduler();
  return 0;
}

int Optimizer::BufQueAlloc(const ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &optimize_graphs) {
  return 0;
}
