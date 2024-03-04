#ifndef ASCENDADAPTER2_CUSTOM_REG_OP_H
#define ASCENDADAPTER2_CUSTOM_REG_OP_H
#include "graph/operator_reg.h"

namespace ge {
// REG_OP(Add)
//    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
//                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
//                           DT_COMPLEX64, DT_STRING}))
//    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
//                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
//                           DT_COMPLEX64, DT_STRING}))
//    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
//                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
//                           DT_COMPLEX64, DT_STRING}))
//    .OP_END_FACTORY_REG(Add)
}

#endif  // ASCENDADAPTER2_CUSTOM_REG_OP_H
