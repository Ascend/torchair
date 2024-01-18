#include <fstream>

#include "codegen.h"

#include "e2e_load_sub_store.h"

int main() {
  ascir::HintGraph test_graph("load_sub_store");
  LoadSubStore_BeforeAutofuse(test_graph);
  LoadSubStore_AfterInferOutput(test_graph);

  std::vector<ascir::ImplGraph> test_impl_graphs = {ascir::ImplGraph("load_sub_store")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadSubStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadSubStore_AfterScheduler(test_impl_graphs[0]);
  LoadSubStore_AfterQueBufAlloc(test_impl_graphs[0]);

  auto codegen = codegen::Codegen(codegen::CodegenOptions{
      .tiling_lib_path = "./libtest_load_sub_store_codegen_tiling_gen.so", .tiling_lib_codegen_symbol = "CodegenTiling"});

  std::fstream kernel_file("load_sub_store_kernel.cpp", std::ios::out);
  std::fstream tiling_file("load_sub_store_tiling.cpp", std::ios::out);
  std::fstream tiling_data_file("load_sub_store_tiling.h", std::ios::out);

  auto result = codegen.Generate(test_graph, test_impl_graphs);
  kernel_file << result.kernel;
  tiling_file << result.tiling;
  tiling_data_file << result.tiling_data;
}
