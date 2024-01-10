#include "codegen.h"

#include <sstream>

#include "codegen_proto.h"
#include "codegen_kernel.h"
#include "codegen_tiling_data.h"

using namespace codegen;

Codegen::Codegen(const CodegenOptions &options)
    : tiling_lib_(options.tiling_lib_path, options.tiling_lib_codegen_symbol) {}

CodegenResult Codegen::Generate(const ascir::HintGraph &graph, const std::vector<ascir::ImplGraph> &impl_graphs) const {
  CodegenResult result;
  result.proto = GenerateProto(graph);
  result.kernel = GenerateKernel(graph, impl_graphs);
  result.tiling_data = GenerateTilingData(graph, impl_graphs);
  result.tiling = GenerateTiling(graph, impl_graphs);
  return result;
}

std::string Codegen::GenerateProto(const ascir::HintGraph &graph) const {
  OpProto op = OpProto::FromGraph(graph);
  nlohmann::json j = op;
  return j.dump();
}

std::string Codegen::GenerateTilingData(const ascir::HintGraph& hint_graph, const std::vector<ascir::ImplGraph>& impl_graphs) const {
  std::stringstream ss;
  for (auto graph : impl_graphs) {
      ss << TilingData(hint_graph.GetName()).Generate(graph);
  }
  return ss.str();
}

std::string Codegen::GenerateTiling(const ascir::HintGraph &graph,
                                    const std::vector<ascir::ImplGraph> &impl_graphs) const {
  return this->tiling_lib_.Generate(graph, impl_graphs);
}

std::string Codegen::GenerateKernel(const ascir::HintGraph &graph, const std::vector<ascir::ImplGraph>& impl_graphs) const {
  // Currently select the first impl graph as the kernel graph.
  ascir::ImplGraph kernel_graph(graph.GetName().c_str());
  kernel_graph.CopyFrom(impl_graphs[0]);

  auto kernel = Kernel::ParseGraph(kernel_graph);
  return kernel.Generate();
}
