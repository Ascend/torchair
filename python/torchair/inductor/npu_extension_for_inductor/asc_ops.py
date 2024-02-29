from torch._inductor.virtualized import V


# TODO: autogen for all ascir ops
def abs(x):
    graph = V.kernel.graph
    op = graph.add_op("Abs")
    op.x = x
    op.y.dtype = x.dtype
    return op.y


def add(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Add")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = x1.dtype
    return op.y


def sub(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Sub")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = x1.dtype
    return op.y


def lt(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Lt")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = x1.dtype
    return op.y


def ge(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Ge")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = x1.dtype
    return op.y


def mul(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Mul")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = x1.dtype
    return op.y


def sigmoid(x):
    graph = V.kernel.graph
    op = graph.add_op("Sigmoid")
    op.x = x
    op.y.dtype = x.dtype
    return op.y


def maximum(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Maximum")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = x1.dtype
    return op.y


def exp(x):
    graph = V.kernel.graph
    op = graph.add_op("Exp")
    op.x = x
    op.y.dtype = x.dtype
    return op.y


def broadcast(x):
    graph = V.kernel.graph
    op = graph.add_op("Broadcast")
    op.x = x
    op.y.dtype = x.dtype
    return op.y


def truediv(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("TrueDiv")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = x1.dtype
    return op.y


def div(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Div")
    op.x1 = x1
    op.x2 = x2
    op.y.dtype = x1.dtype
    return op.y


def cast(x, *, dst, src=None):
    graph = V.kernel.graph
    op = graph.add_op("Cast")
    op.x = x
    op.dst_type = dst
    op.y.dtype = dst
    return op.y


def reduction(x, *, reduce_type):
    graph = V.kernel.graph
    op = graph.add_op(reduce_type.capitalize())
    op.x = x
    op.attr.hint.compute_type = 'reduce'
    op.y.dtype = x.dtype
    return op.y


def data(*, name, input=None, sizes=(), dtype):
    graph = V.kernel.graph
    op = graph.add_op("Data", name=name)
    op.y.size = sizes
    op.y.dtype = dtype
    if input:
        op.x = input
    return op.y


def output(*, name, input=None, sizes=(), dtype):
    graph = V.kernel.graph
    op = graph.add_op("Output", name=name)
    op.y.size = sizes
    op.y.dtype = dtype
    if input:
        op.x = input
    return op.y


def workspace(*, name, input=None, sizes=(), dtype):
    graph = V.kernel.graph
    op = graph.add_op("Workspace", name=name)
    op.y.size = sizes
    op.y.dtype = dtype
    if input:
        op.x = input
    return op.y


def load(data):
    graph = V.kernel.graph
    op = graph.add_op("Load")
    op.x = data
    op.y.dtype = data.dtype
    return op.y


def store(value):
    graph = V.kernel.graph
    op = graph.add_op("Store")
    op.x = value
    op.y.dtype = value.dtype
    return op.y