from typing import List

import sympy


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


class Loop:
    def __init__(self):
        self.axis: List[sympy.Symbol] = []
        self.stride: List[sympy.Expr] = []
        self.size: List[sympy.Expr] = []
        self.offset: sympy.Expr = sympy.Symbol("0")

    @property
    def asc_offset(self):
        return f"[{AscExpr(self.offset)}]"

    @property
    def asc_axis(self):
        axis = ', '.join([f"{s.name}" for s in self.axis])
        return f"[{axis}]"

    @property
    def asc_stride(self):
        return f"[{', '.join([str(AscExpr(exp)) for exp in self.stride])}]"

    @property
    def asc_size(self):
        return f"[{', '.join([str(AscExpr(exp)) for exp in self.size])}]"
