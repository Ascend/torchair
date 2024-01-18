#include "ascir.h"

#include <sstream>

extern "C" std::string CodegenTiling(const std::vector<ascir::ImplGraph> &impl_graphs) {
  std::stringstream ss;

  ss << "extern \"C\" void GetTiling(optiling::TilingData& tiling_data) {" << std::endl;
  ss << "  tiling_data.set_block_dim(48);" << std::endl;
  ss << "  tiling_data.set_z0z1Tb_size(tiling_data.get_s0() * tiling_data.get_s1() / 4 / 48);" << std::endl;
  ss << "  tiling_data.set_z1t_size(4);" << std::endl;
  ss << "}";

  return ss.str();
}

