#ifndef __AUTOFUSE_CODEGEN_H__
#define __AUTOFUSE_CODEGEN_H__

#include "ascir.h"

namespace codegen {
struct CodegenResult {
  std::string proto;
  std::string tiling_data;
  std::string tiling;
  std::string kernel;
};

struct CodegenOptions {};

class Codegen {
 public:
  explicit Codegen(const CodegenOptions& options);
  CodegenResult Generate(const ascir::HintGraph &graph, const std::vector<ascir::ImplGraph>& impl_graphs) const;

  std::string GenerateProto(const ascir::HintGraph& graph) const;
  std::string GenerateTilingData(const std::vector<ascir::ImplGraph>& impl_graphs) const;
  std::string GenerateTiling(const std::vector<ascir::ImplGraph>& impl_graphs) const;
  std::string GenerateKernel(const std::vector<ascir::ImplGraph>& impl_graphs) const;
};
} // namespace codegen

#endif

