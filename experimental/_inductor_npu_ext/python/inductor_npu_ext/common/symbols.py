import functools
import operator
from typing import List

import sympy

from torch._inductor.virtualized import V
from inductor_npu_ext.common.utils import StrRep


class AscExpr:
    def __init__(self, expr: sympy.Expr):
        if not isinstance(expr, (sympy.Expr, sympy.Symbol)):
            expr = sympy.Symbol(str(expr))
        self.expr = sympy.simplify(expr)

    def __str__(self):
        return str(self.expr).replace('//', '/')

    def __repr__(self):
        return repr(self.expr).replace('//', '/')


class Loop:
    def __init__(self, *, axis=None, size=None, stride=None, offset=None):
        self.axis: List[sympy.Symbol] = axis if axis is not None else []
        self.size: List[sympy.Expr] = size if size is not None else []
        self.stride: List[sympy.Expr] = stride if stride is not None else []
        self.offset: sympy.Expr = offset if offset is not None else sympy.Symbol("0")

    def __str__(self):
        return f"{self.axis}|{self.size}|{self.stride}|{self.offset}"

    def __eq__(self, other: 'Loop') -> bool:
        return str(self.axis) == str(other.axis) and str(self.size) == str(other.size) and \
            str(self.stride) == str(other.stride) and str(self.offset) == str(other.offset)

    @staticmethod
    def get_hint(sym):
        try:
            return V.graph.sizevars.size_hint(sym)
        except Exception:
            return sym

    @property
    def hint_size(self):
        return [self.get_hint(x) for x in self.size]

    @property
    def hint_stride(self):
        return [self.get_hint(x) for x in self.stride]

    @property
    def hint_offset(self):
        return self.get_hint(self.offset)

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

    def is_contiguous(self):
        if len(self.axis) == 0:
            return True
        return [str(v) for v in self.stride] == [str(v) for v in self.contiguous_stride()]

    def contiguous_stride(self):
        stride = []
        multiplier = sympy.S.One
        for s in reversed(self.size):
            stride.append(V.graph.sizevars.simplify(multiplier) if str(s) != "1" else sympy.S.Zero)
            multiplier *= s

        return list(reversed(stride))

    def contiguous_(self, zero_offset=False):
        if zero_offset:
            self.offset = sympy.S.Zero
        if len(self.axis) == 0:
            return self

        self.stride = self.contiguous_stride()
        return self

    def transpose_(self, dim0, dim1):
        def swap(swapped, i, j):
            swapped[i], swapped[j] = swapped[j], swapped[i]

        swap(self.axis, dim0, dim1)
        swap(self.stride, dim0, dim1)
        swap(self.size, dim0, dim1)
        return self

    def broadcast_(self, dim, size):
        self.size[dim] = size
        self.stride[:dim] = [size * stride for stride in self.stride[:dim]]
        self.stride[dim] = functools.reduce(operator.mul, self.size[dim + 1:], sympy.S.One)

    def debroadcast_(self, dim):
        size = self.size[dim]
        self.size[dim] = sympy.S.One
        self.stride[:dim] = [stride // size for stride in self.stride[:dim]]
        self.stride[dim] = sympy.S.Zero
        return self

    def zero_offset_(self):
        offset = self.offset
        self.offset = sympy.S.Zero
        return offset

    def copy(self):
        return Loop(axis=self.axis.copy(), size=self.size.copy(), stride=self.stride.copy(), offset=self.offset)

    def encode(self):
        size_str = "_".join([str(s) for s in self.size])
        stride_str = "_".join([str(s) for s in self.stride])
        return f"shape{size_str}stride{stride_str}offset{self.offset}"

    def codegen_contiguous(self, name):
        size = f"{tuple(self.size)}"
        stride = f"{tuple(self.stride)}"
        return f"reinterpret_tensor({name}, {size}, {stride}, {self.offset}).contiguous()"


class DenseLoop(Loop):
    def __init__(self, axis, size):
        super().__init__(axis=axis, size=size)
        self.contiguous_()


class Axis:
    def __init__(self, name, size, order):
        self.name = name
        self.size = size
        self.order = order

    def __lt__(self, other):
        return False

    def __str__(self):
        return f"{self.name}({self.size})"

    def __repr__(self):
        return f"{self.name}({self.size})"
