#include "e2e_load_sub_store.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "ascir_utils.h"
#include "optimize.h"
#include "codegen.h"

class E2E_LoadSubStore : public ::testing::Test {
protected:
    optimize::Optimizer optimizer;
    codegen::Codegen codegen;

    E2E_LoadSubStore() : optimizer(optimize::OptimizerOptions{}), codegen(codegen::CodegenOptions{}) {}
};
