from typing import List

import torch
from torch._inductor.virtualized import V
from inductor_npu_ext.common.asc_graph import _Tensor, _Op, ASCGraph


class _AscOpWrapper:
    def __init__(self, f):
        self._op = f.__name__
        self._f = f

    def _set_loop(self, tensors, loop=None):
        if self._op in ["data", "output", "workspace"]:
            return
        loop = loop if loop else V.kernel.contiguous_loop
        tensors: List[_Tensor] = tensors if isinstance(tensors, (list, tuple)) else [tensors]
        for tensor in tensors:
            tensor.as_loop(loop)

    def __call__(self, *args, loop=None, **kwargs):
        graph: ASCGraph = V.kernel.graph
        cache_key = f"{self._op}({args},{loop},{kwargs})"
        if cache_key in graph.op_cache:
            return graph.op_cache[cache_key]
        tensors = self._f(*args, **kwargs)
        if isinstance(tensors, (list, tuple)):
            tensors = (_Tensor(t) if not isinstance(t, _Tensor) else t for t in tensors)
            self._set_loop(tensors, loop)
            graph.op_cache[cache_key] = tensors
            return tensors
        tensor = _Tensor(tensors) if not isinstance(tensors, _Tensor) else tensors
        self._set_loop(tensor, loop)
        graph.op_cache[cache_key] = tensor
        return tensor


def asc_ops(fn):
    def wrapper(*args, **kwargs):
        return _AscOpWrapper(fn)(*args, **kwargs)
    return wrapper


def Op(t) -> _Op:
    return V.kernel.graph.add_op(t)


@asc_ops
def abs(x):
    op = Op("Abs")
    op.x = x
    return op.y


@asc_ops
def add(x1, x2):
    op = Op("Add")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def sub(x1, x2):
    op = Op("Sub")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def ge(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Ge")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def gt(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Gt")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def le(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Le")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def lt(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Lt")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def eq(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Eq")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def ne(x1, x2):
    x1 = cast(x1, dst=torch.float32)
    x2 = cast(x2, dst=torch.float32)
    op = Op("Ne")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def mul(x1, x2):
    op = Op("Mul")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def sigmoid(x):
    op = Op("Sigmoid")
    op.x = x
    return op.y


@asc_ops
def sqrt(x):
    op = Op("Sqrt")
    op.x = x
    return op.y


@asc_ops
def log(x):
    op = Op("Ln")
    op.x = x
    return op.y


@asc_ops
def rsqrt(x):
    op = Op("Rsqrt")
    op.x = x
    return op.y


@asc_ops
def maximum(x1, x2):
    op = Op("Maximum")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def exp(x):
    op = Op("Exp")
    op.x = x
    return op.y


@asc_ops
def broadcast(x):
    op = Op("Broadcast")
    op.x = x
    return op.y


@asc_ops
def transpose(x):
    op = Op("Transpose")
    op.x = x
    return op.y


@asc_ops
def reciprocal(x):
    op = Op("Reciprocal")
    op.x = x
    return op.y


@asc_ops
def truediv(x1, x2):
    if x1 == '1':
        return reciprocal(x2)
    op = Op("TrueDiv")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def div(x1, x2):
    return truediv(x1, x2)


@asc_ops
def square(x):
    return mul(x, x)


@asc_ops
def cast(x, *, dst, src=None, use_compute_types=True):
    op = Op("Cast")
    op.x = x
    op.y.dtype = dst
    return op.y


@asc_ops
def constant(value: str, dtype):
    op = Op("Scalar")
    op.attr.ir_attr.value = value
    op.y.dtype = dtype
    return op.y


@asc_ops
def reduction(x, *, reduce_type):
    op = Op(reduce_type.capitalize())
    op.x = x
    op.set_private_attr("compute_type", "reduce")
    return op.y


@asc_ops
def data(*, name, dtype, index):
    # drop name for stable cache hint
    op = Op("Data")
    op.attr.ir_attr.index = index
    op.y.dtype = dtype
    return op.y


@asc_ops
def output(*, name, src, dtype, index):
    # drop name for stable cache hint
    op = Op("Output")
    op.attr.ir_attr.index = index
    op.x = src
    op.y.dtype = dtype
    return op.y


@asc_ops
def workspace(*, name, src, dtype):
    op = Op("Workspace", name=name)
    op.x = src
    op.y.dtype = dtype
    return op.y


@asc_ops
def load(buffer, *, offset):
    op = Op("Load")
    if offset is not None:
        op.attr.ir_attr.offset = offset
    op.x = buffer
    return op.y


@asc_ops
def store(value):
    op = Op("Store")
    op.x = value
    return op.y


@asc_ops
def bitwise_and(x1, x2):
    op = Op("BitwiseAnd")
    op.x1 = x1
    op.x2 = x2
    return op.y


@asc_ops
def where(x1, x2, x3):
    op = Op("Where")
    op.x1 = x1
    op.x2 = x2
    op.x3 = x3
    return op.y
