from torch._inductor.virtualized import V
from npu_extension_for_inductor.common.utils import TypeUtils
import torch


# TODO: autogen for all ascir ops
def abs(x):
    graph = V.kernel.graph
    op = graph.add_op("Abs")
    op.x = x
    return op.y


def add(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Add")
    op.x1 = x1
    op.x2 = x2
    return op.y


def sub(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Sub")
    op.x1 = x1
    op.x2 = x2
    return op.y


def lt(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Lt")
    op.x1 = x1
    op.x2 = x2
    return op.y


def ge(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Ge")
    op.x1 = x1
    op.x2 = x2
    return op.y


def gt(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Gt")
    op.x1 = x1
    op.x2 = x2
    return op.y


def mul(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Mul")
    op.x1 = x1
    op.x2 = x2
    return op.y


def sigmoid(x):
    graph = V.kernel.graph
    op = graph.add_op("Sigmoid")
    op.x = x
    return op.y


def sqrt(x):
    graph = V.kernel.graph
    op = graph.add_op("Sqrt")
    op.x = x
    return op.y


def rsqrt(x):
    graph = V.kernel.graph
    op = graph.add_op("Rsqrt")
    op.x = x
    return op.y


def maximum(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Maximum")
    op.x1 = x1
    op.x2 = x2
    return op.y


def exp(x):
    graph = V.kernel.graph
    op = graph.add_op("Exp")
    op.x = x
    return op.y


def broadcast(x):
    graph = V.kernel.graph
    op = graph.add_op("Broadcast")
    op.x = x
    return op.y


def truediv(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("TrueDiv")
    op.x1 = x1
    op.x2 = x2
    return op.y


def div(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Div")
    op.x1 = x1
    op.x2 = x2
    return op.y


def cast(x, *, dst, src=None):
    graph = V.kernel.graph
    op = graph.add_op("Cast")
    op.x = x
    op.y.dtype = dst
    return op.y


def concat(*x):
    graph = V.kernel.graph
    op = graph.add_op("Concat")
    op.x = list(x)
    return op.y


def constant(value: str, dtype):
    graph = V.kernel.graph
    op = graph.add_op("Scalar")
    op.attr.ir_attr.value = value
    op.y.dtype = dtype
    return op.y


def reduction(x, *, reduce_type):
    graph = V.kernel.graph
    op = graph.add_op(reduce_type.capitalize())
    op.x = x
    op.set_private_attr("compute_type", "reduce")
    return op.y


def data(*, name, dtype, index):
    graph = V.kernel.graph
    op = graph.add_op("Data", name=name)
    op.attr.ir_attr.index = index
    op.y.dtype = dtype
    return op.y


def output(*, name, src, dtype, index):
    graph = V.kernel.graph
    op = graph.add_op("Output", name=name)
    op.attr.ir_attr.index = index
    op.x = src
    op.y.dtype = dtype
    return op.y


def workspace(*, name, src, dtype):
    graph = V.kernel.graph
    op = graph.add_op("Workspace", name=name)
    op.x = src
    op.y.dtype = dtype
    return op.y


def load(buffer, *, offset):
    graph = V.kernel.graph
    op = graph.add_op("Load")
    if offset is not None:
        op.attr.ir_attr.offset = offset
    op.x = buffer
    return op.y


def store(value):
    graph = V.kernel.graph
    op = graph.add_op("Store")
    op.x = value
    return op.y
