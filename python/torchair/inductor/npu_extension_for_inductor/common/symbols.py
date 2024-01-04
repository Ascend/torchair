from typing import List

import sympy
from npu_extension_for_inductor.common.utils import StrRep


class AscSymbol:
    def __init__(self, s):
        self.s = [s] if isinstance(s, str) else s

    def __mul__(self, other):
        if isinstance(other, AscSymbol):
            return AscSymbol(self.s + other.s)
        return AscSymbol(self.s + [other])

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power, modulo=None):
        return AscSymbol(self.s * power)

    def __str__(self):
        multipliers = [str(s) for s in self.s if str(s) != "1"]
        if len(multipliers) == 0:
            return "1"
        return "*".join(multipliers)

    def __repr__(self):
        multipliers = [f"{s}" for s in self.s if str(s) != "1"]
        return f"ascir.SizeExpr([{','.join(multipliers)}])"


class AscExpr:
    def __init__(self, expr: sympy.Expr):
        self.expr = expr
        vals = dict([(str(symbol), AscSymbol(str(symbol))) for symbol in self.expr.free_symbols])
        self.asc_expr = eval(str(self.expr), vals)
        if not isinstance(self.asc_expr, AscSymbol):
            self.asc_expr = AscSymbol(str(self.asc_expr))

    def __str__(self):
        return str(self.asc_expr)

    def __repr__(self):
        return repr(self.asc_expr)


class Loop:
    def __init__(self):
        self.axis: List[sympy.Symbol] = []
        self.stride: List[sympy.Expr] = []
        self.size: List[sympy.Expr] = []
        self.offset: sympy.Expr = sympy.Symbol("0")

    def is_contiguous(self):
        if len(self.axis) == 0:
            return True
        if str(self.stride[-1]) != "1":
            return False
        for i in range(len(self.axis) - 1):
            if self.stride[i] != self.size[i + 1]:
                return False
        return True

    def contiguous(self):
        if len(self.axis) == 0:
            return self
        self.stride[-1] = sympy.Symbol("1")
        for i in range(len(self.axis) - 1):
            self.stride[i] = self.size[i + 1]
        return self

    @property
    def asc_offset(self):
        return AscExpr(self.offset)

    @property
    def asc_axis(self):
        return [StrRep(s.name) for s in self.axis]

    @property
    def asc_stride(self):
        return [AscExpr(exp) for exp in self.stride]

    @property
    def asc_size(self):
        return [AscExpr(exp) for exp in self.size]