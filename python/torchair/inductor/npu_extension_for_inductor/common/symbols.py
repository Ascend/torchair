import operator
from typing import List

import sympy
from npu_extension_for_inductor.common.utils import StrRep


class AscSymbol:
    @classmethod
    def _as_symbol(cls, v):
        if not isinstance(v, AscSymbol):
            return AscSymbol(str(v))
        return v

    def __init__(self, name=None, *, operands=None):
        if operands is not None:
            assert isinstance(operands, (list, tuple))
            self.operands = operands
        else:
            assert isinstance(name, str)
            self.sym = sympy.Symbol(name)
            self.operands = [self]

    @property
    def name(self):
        assert self.is_operand(), f"Only operand has name, got {self}"
        return self.sym.name

    def free_operands(self):
        if self.is_operand():
            return [self]
        operands = []
        for operand in self.operands:
            operands.extend(operand.free_operands())
        return operands

    def operators(self, recursive=False):
        operators = set()
        if isinstance(self, Asc2OpeSymbol):
            operators.add(self.op)
            if recursive:
                for operand in self.operands:
                    operators.update(operand.operators(recursive=True))
        return operators

    def __mul__(self, other):
        return Asc2OpeSymbol(self, AscSymbol._as_symbol(other), operator.mul)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return Asc2OpeSymbol(self, AscSymbol._as_symbol(other), operator.add)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return Asc2OpeSymbol(self, AscSymbol._as_symbol(other), operator.sub)

    def __rsub__(self, other):
        return self - other

    def __truediv__(self, other):
        return Asc2OpeSymbol(self, AscSymbol._as_symbol(other), operator.truediv)

    def __rtruediv__(self, other):
        return self / other

    def __floordiv__(self, other):
        return Asc2OpeSymbol(self, AscSymbol._as_symbol(other), operator.floordiv)

    def __rfloordiv__(self, other):
        return self // other

    def __mod__(self, other):
        return Asc2OpeSymbol(self, AscSymbol._as_symbol(other), operator.mod)

    def __rmod__(self, other):
        return self % other

    def __pow__(self, power, modulo=None):
        pow_symbol = self
        for i in range(power - 1):
            pow_symbol *= self
        return pow_symbol

    def is_operand(self):
        return len(self.operands) == 1 and self.operands[0] is self

    def is_operator(self, op, recursive=False):
        if isinstance(self, Asc2OpeSymbol) and self.op == op:
            if not recursive:
                return True
            return all([operand.is_operator(op, recursive=True) or operand.is_operand() for operand in self.operands])
        return False

    def is_mul(self, recursive=False):
        return self.is_operator(operator.mul, recursive)

    def is_add(self, recursive=False):
        return self.is_operator(operator.add, recursive)

    def is_sub(self, recursive=False):
        return self.is_operator(operator.sub, recursive)

    def is_truediv(self, recursive=False):
        return self.is_operator(operator.truediv, recursive)

    def is_floordiv(self, recursive=False):
        return self.is_operator(operator.floordiv, recursive)

    def is_mod(self, recursive=False):
        return self.is_operator(operator.mod, recursive)

    def is_pow(self, recursive=False):
        return self.is_operator(operator.pow, recursive)

    def __str__(self):
        return str(self.sym)

    def __repr__(self):
        return f"ascir.SizeExpr([{'' if self.sym.name == '1' else self.sym.name}])"


class Asc2OpeSymbol(AscSymbol):
    def __init__(self, left, right, op):
        super().__init__(operands=[left, right])
        self.op = op
        self.sym = self.op(left.sym, right.sym)

    @property
    def x(self):
        return self.operands[0]

    @property
    def y(self):
        return self.operands[1]

    def __str__(self):
        return str(self.sym)

    def __repr__(self):
        if self.is_mul(recursive=True):
            return f"ascir.SizeExpr([{','.join([v.name for v in self.free_operands() if v.name != '1'])}])"
        subs = dict()
        for symbol in self.sym.free_symbols:
            subs[symbol] = sympy.Symbol(f"ascir.SizeExpr([{symbol}])")
        return str(self.sym.subs(subs))


class AscExpr:
    def __init__(self, expr: sympy.Expr):
        if not isinstance(expr, (sympy.Expr, sympy.Symbol)):
            expr = sympy.Symbol(str(expr))
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

    def contiguous_(self):
        if len(self.axis) == 0:
            return self
        self.stride = list([None] * len(self.axis))
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
