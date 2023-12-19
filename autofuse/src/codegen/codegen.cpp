#include "codegen.h"

using namespace codegen;

Codegen::Codegen(const CodegenOptions &options) {}

CodegenResult Codegen::Generate(const ascir::HintGraph &graph, const std::vector<ascir::ImplGraph> &impl_graphs) const {
  CodegenResult result;
  return result;
}

std::string Codegen::GenerateProto(const ascir::HintGraph &graph) const {
  return "";
}

std::string Codegen::GenerateTilingData(const std::vector<ascir::ImplGraph>& impl_graphs) const {
  return "";
}

std::string Codegen::GenerateTiling(const std::vector<ascir::ImplGraph>& impl_graphs) const {
  return "";
}

std::string Codegen::GenerateKernel(const std::vector<ascir::ImplGraph>& impl_graphs) const {
  return "";
}
