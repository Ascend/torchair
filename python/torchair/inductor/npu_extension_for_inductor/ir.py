from torch._inductor.virtualized import V


# TODO: autogen for all ascir ops
def abs(x):
    graph = V.kernel.graph
    op = graph.add_op("Abs")
    graph.buffer.writeline(f"{op}.x = {x}")
    return f"{op}.y"


def add(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Add")
    graph.buffer.writeline(f"{op}.x1 = {x1}")
    graph.buffer.writeline(f"{op}.x2 = {x2}")
    return f"{op}.y"


def sub(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("Sub")
    graph.buffer.writeline(f"{op}.x1 = {x1}")
    graph.buffer.writeline(f"{op}.x2 = {x2}")
    return f"{op}.y"


def exp(x):
    graph = V.kernel.graph
    op = graph.add_op("Exp")
    graph.buffer.writeline(f"{op}.x = {x}")
    return f"{op}.y"


def broadcast(x):
    graph = V.kernel.graph
    op = graph.add_op("Broadcast")
    graph.buffer.writeline(f"{op}.x = {x}")
    return f"{op}.y"


def truediv(x1, x2):
    graph = V.kernel.graph
    op = graph.add_op("TrueDiv")
    graph.buffer.writeline(f"{op}.x1 = {x1}")
    graph.buffer.writeline(f"{op}.x2 = {x2}")
    return f"{op}.y"


def to_dtype(x, dst_dtype, src_dtype=None):
    graph = V.kernel.graph
    op = graph.add_op("ToDtype")
    graph.buffer.writeline(f"{op}.x = {x}")
    graph.buffer.writeline(f"{op}.attr.src_dtype = {src_dtype}")
    graph.buffer.writeline(f"{op}.attr.dst_dtype = {dst_dtype}")
    return f"{op}.y"


def reduction(x, *, dst_dtype, src_dtype, reduce_type):
    graph = V.kernel.graph
    op = graph.add_op(reduce_type.capitalize())
    graph.buffer.writeline(f"{op}.x = {x}")
    graph.buffer.writeline(f"{op}.attr.src_dtype = {src_dtype}")
    graph.buffer.writeline(f"{op}.attr.dst_dtype = {dst_dtype}")
    graph.buffer.writeline(f"{op}.attr.compute_type = 'reduce'")
    return f"{op}.y"


def data(name, *, input=None, sizes=(), dtype=None):
    graph = V.kernel.graph
    op = graph.add_op("Data", name)
    graph.buffer.writeline(f"{op}.y.size = [{', '.join(sizes)}]")
    graph.buffer.writeline(f"{op}.y.dtype = {dtype}")
    if input:
        graph.buffer.writeline(f"{op}.x = {input}")
    return f"{op}.y"


def load(data):
    graph = V.kernel.graph
    op = graph.add_op("Load")
    graph.buffer.writeline(f"{op}.x = {data}")
    return f"{op}.y"


def store(value):
    graph = V.kernel.graph
    op = graph.add_op("Store")
    graph.buffer.writeline(f"{op}.x = {value}")
    return f"{op}.y"