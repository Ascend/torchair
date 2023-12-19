#include "gtest/gtest.h"

#include "ascir_ops.h"

#include "graph/operator_reg.h"

namespace ge {
REG_OP(TestOp).INPUT(x, BasicType()).OP_END_FACTORY_REG(TestOp);
};

TEST(AscirOps_Register, RegOp_WillCreateAscirOpFactory) {
  ge::op::TestOp op("test_op");

  ascir::Graph graph("test");
  graph.SetInputs({op});
}
