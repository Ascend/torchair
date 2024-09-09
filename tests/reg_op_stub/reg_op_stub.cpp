#include "graph/operator_reg.h"
#include "graph/operator_factory.h"

using namespace ge;

namespace ge {
REG_OP(TestV1)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .DYNAMIC_OUTPUT(y2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .REQUIRED_ATTR(attr1, Int)
    .ATTR(attr2, Float, 1.0)
    .OP_END_FACTORY_REG(TestV1);

REG_OP(TestV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x4, TensorType({DT_FLOAT16, DT_BOOL, DT_FLOAT32}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .DYNAMIC_OUTPUT(y2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .REQUIRED_ATTR(attr1, Int)
    .ATTR(attr2, Float, 1.0)
    .ATTR(attr3, String, "BSH")
    .OP_END_FACTORY_REG(TestV2);
}
