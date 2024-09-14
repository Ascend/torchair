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

REG_OP(MatmulAllReduce)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT64, DT_INT64}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(comm_quant_scale_1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(comm_quant_scale_2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .OP_END_FACTORY_REG(MatmulAllReduce);
}
