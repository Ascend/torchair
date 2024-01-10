#ifndef __AUTOFUSE_CODEGEN_H__
#define __AUTOFUSE_CODEGEN_H__

#include "ascir.h"
#include "codegen_tiling.h"

namespace codegen {
struct CodegenResult {
  std::string proto;
  std::string tiling_data;
  std::string tiling;
  std::string kernel;
};

struct CodegenOptions {
  std::string tiling_lib_path;
  std::string tiling_lib_codegen_symbol;
};

class Codegen {
 public:
  explicit Codegen(const CodegenOptions& options);
  CodegenResult Generate(const ascir::HintGraph &graph, const std::vector<ascir::ImplGraph>& impl_graphs) const;

  std::string GenerateProto(const ascir::HintGraph& graph) const;
  std::string GenerateTilingData(const ascir::HintGraph& graph, const std::vector<ascir::ImplGraph>& impl_graphs) const;
  std::string GenerateTiling(const ascir::HintGraph &graph, const std::vector<ascir::ImplGraph> &impl_graphs) const;
  std::string GenerateKernel(const ascir::HintGraph &graph, const std::vector<ascir::ImplGraph> &impl_graphs) const;

 private:
  TilingLib tiling_lib_;
};
} // namespace codegen

#endif

