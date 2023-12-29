#include "codegen.h"

#include <sstream>

#include "codegen_proto.h"
#include "codegen_kernel.h"

using namespace codegen;

Codegen::Codegen(const CodegenOptions &options) {}

CodegenResult Codegen::Generate(const ascir::HintGraph &graph, const std::vector<ascir::ImplGraph> &impl_graphs) const {
  CodegenResult result;
  return result;
}

std::string Codegen::GenerateProto(const ascir::HintGraph &graph) const {
  OpProto op = OpProto::FromGraph(graph);
  nlohmann::json j;
  j.push_back(op);
  return j.dump();
}

std::string Codegen::GenerateTilingData(const std::vector<ascir::ImplGraph>& impl_graphs) const {
  return "";
}

std::string Codegen::GenerateTiling(const std::vector<ascir::ImplGraph>& impl_graphs) const {
  return "";
}

std::string Codegen::GenerateKernel(const std::vector<ascir::ImplGraph>& impl_graphs) const {
  std::stringstream ss;

  for (auto &impl_graph : impl_graphs) {
    auto kernel = Kernel::ParseGraph(impl_graph);
    ss << kernel.Generate();
  }

  return ss.str();
}
