#include "e2e_softmax.h"

#include "gtest/gtest.h"

#include "optimize.h"
#include "codegen.h"
#include "ascir_utils.h"

class E2E_Softmax : public ::testing::Test {
 protected:
  optimize::Optimizer optimizer;
  codegen::Codegen codegen;

  E2E_Softmax() : optimizer(optimize::OptimizerOptions{}), codegen(codegen::CodegenOptions{}) {}
};

TEST_F(E2E_Softmax, ConstructGraphWithAscir) {
  ascir::HintGraph graph("graph");
  Softmax_BeforeAutofuse(graph);

  std::cout << ascir::utils::DebugStr(graph) << std::endl;
}
