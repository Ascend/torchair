#ifndef __CODEGEN_TILING_H__
#define __CODEGEN_TILING_H__

#include "ascir.h"

namespace codegen {
  using TilingLibCodegenFunc = std::string (*)(const std::vector<ascir::ImplGraph>& graphs);

  class TilingLib {
   public:
    TilingLib(const std::string &lib_path, const std::string &codegen_symbol_name);
    std::string Generate(const ascir::HintGraph &graph, const std::vector<ascir::ImplGraph> &graphs) const;

   protected:
    std::string TilingFuncDef(const ascir::HintGraph &graph) const;
    std::string InferShapeDef(const ascir::HintGraph &graph) const;
    std::string OpDef(const ascir::HintGraph &graph) const;

    std::string OpInputDef(const ascir::NodeView& node) const;
    std::string OpOutputDef(const ascir::NodeView& node) const;

   private:
    TilingLibCodegenFunc codegen_func_;
  };
};

#endif
