#include <fstream>

#include "ascir.h"
#include "ascir_utils.h"
#include "ascir_ops.h"
#include "optimize.h"
#include "codegen.h"

#include "e2e_broadcast.h"

using namespace ascir;
using namespace ascir::ops;

int main(int argc, char *argv[]) {
  Graph graph("LoadBroadcastStore");
  LoadBroadcastStore_BeforeAutofuse(graph);

  Graph impl_graph("LoadBroadcastStore_1");
  impl_graph.CopyFrom(graph);
  LoadBroadcastStore_AfterAutofuse(impl_graph);

  std::cout << ascir::utils::DebugImplGraphStr(impl_graph) << std::endl;

  codegen::Codegen c(codegen::CodegenOptions{
      .tiling_lib_path = "./libtest_load_broadcast_store_codegen_tiling_gen.so",
      .tiling_lib_codegen_symbol = "CodegenTiling",
  });
  auto result = c.Generate(graph, {impl_graph});

  std::fstream kernel_file("load_broadcast_store_kernel.cpp", std::ios::out);
  std::fstream tiling_file("load_broadcast_store_tiling.cpp", std::ios::out);
  std::fstream tiling_data_file("load_broadcast_store_tiling.h", std::ios::out);

  kernel_file << result.kernel;
  tiling_file << result.tiling;
  tiling_data_file << result.tiling_data;

  return 0;
}
