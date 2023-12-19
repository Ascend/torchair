#ifndef __ASCIR_OPS_H__
#define __ASCIR_OPS_H__

#include "ascir.h"

REG_OPS(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .END_OPS(Data)

namespace ascir::ops {
  struct Data : public ge::op::Data {
    union {
      ge::Operator *__op;
      ascir::OperatorInput<0> x;
      ascir::OperatorOutput<0> y;
    };

    static constexpr const char *Type = "Data";
    ascir::NodeAttr attr;

    inline Data(const char *name) : ge::op::Data(name), __op(this), attr(*this) {}
  };
};

REG_OPS(Load)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .END_OPS(Load);

namespace ascir::ops {
struct Load : public ge::op::Load {
  union {
    ge::Operator *__op;
    ascir::OperatorInput<0> x;
    ascir::OperatorOutput<0> y;
  };

  static constexpr const char *Type = "Load";
  ascir::NodeAttr attr;

  inline Load(const char *name) : ge::op::Load(name), __op(this), attr(*this) {}
};
}  // namespace ascir::ops

REG_OPS(Abs)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .END_OPS(Abs);

namespace ascir::ops {
struct Abs : public ge::op::Abs {
  union {
    ge::Operator *__op;
    ascir::OperatorInput<0> x;
    ascir::OperatorOutput<0> y;
  };

  static constexpr const char *Type = "Abs";
  ascir::NodeAttr attr;

  inline Abs(const char *name) : ge::op::Abs(name), __op(this), attr(*this) {}
};
}  // namespace ascir::ops

REG_OPS(Store)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .END_OPS(Store)

namespace ascir::ops {
struct Store : public ge::op::Store {
  union {
    ge::Operator *__op;
    ascir::OperatorInput<0> x;
    ascir::OperatorOutput<0> y;
  };

  static constexpr const char *Type = "Store";
  NodeAttr attr;
  inline Store(const char *name) : ge::op::Store(name), __op(this), attr(*this) {}
};
}  // namespace ascir::ops

#endif
