#ifndef __ASCIR_OPS_H__
#define __ASCIR_OPS_H__

#include "ascir.h"

namespace ascir::ops {
template <typename T>
bool IsOps(const ascir::NodeView &view) {
  return view->GetType() == T::Type;
}
};

namespace ge {
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Data)
}

namespace ascir::ops {
REG_OPS(Data)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Data)
}

namespace ge {
REG_OP(Workspace)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Workspace)
}

namespace ascir::ops {
REG_OPS(Workspace)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Workspace)
}

namespace ge {
REG_OP(Output)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Output)
}

namespace ascir::ops {
REG_OPS(Output)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Output)
}

namespace ge {
REG_OP(Load)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Load);
}

namespace ascir::ops {
REG_OPS(Load)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Load)
}  // namespace ascir::ops

namespace ge {
REG_OP(Broadcast)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Broadcast);
}

namespace ascir::ops {
REG_OPS(Broadcast)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Broadcast)
}  // namespace ascir::ops

namespace ge {
REG_OP(Store)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Store)
}

namespace ascir::ops {
REG_OPS(Store)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Store)
}  // namespace ascir::ops

namespace ge {
REG_OP(Nop)
    .REQUIRED_ATTR(dst_type, Int)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Nop)
}

namespace ascir::ops {
REG_OPS(Nop)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Nop)
}

namespace ge {
REG_OP(Cast)
    .REQUIRED_ATTR(dst_type, Int)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Cast)
}

namespace ascir::ops {
REG_OPS_WITH_ATTR(Cast)
  OPS_ATTR_NAME_START()
    OPS_ATTR_NAME(dst_type)
  OPS_ATTR_NAME_END()
    OPS_ATTR(dst_type, ge::DataType)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Cast)
}

namespace ge {
REG_OP(Abs)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Abs);
}

namespace ascir::ops {
REG_OPS(Abs)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Abs)
}  // namespace ascir::ops

namespace ge {
REG_OP(Max)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Max);
}

namespace ascir::ops {
REG_OPS(Max)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Max)
}  // namespace ascir::ops

namespace ge {
REG_OP(Sum)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Sum);
}

namespace ascir::ops {
REG_OPS(Sum)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Sum)
}  // namespace ascir::ops

namespace ge {
REG_OP(Sub)
    .INPUT(x1, TensorType::ALL())
    .INPUT(x2, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Sub);
}

namespace ascir::ops {
REG_OPS(Sub)
  OPS_INPUT(0, x1)
  OPS_INPUT(1, x2)
  OPS_OUTPUT(0, y)
END_OPS(Sub)
}  // namespace ascir::ops

namespace ge {
REG_OP(Div)
    .INPUT(x1, TensorType::ALL())
    .INPUT(x2, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Div);
}

namespace ascir::ops {
REG_OPS(Div)
  OPS_INPUT(0, x1)
  OPS_INPUT(1, x2)
  OPS_OUTPUT(0, y)
END_OPS(Div)
}  // namespace ascir::ops

namespace ge {
REG_OP(Exp)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Exp);
}

namespace ascir::ops {
REG_OPS(Exp)
  OPS_INPUT(0, x)
  OPS_OUTPUT(0, y)
END_OPS(Exp)
}  // namespace ascir::ops

#endif
