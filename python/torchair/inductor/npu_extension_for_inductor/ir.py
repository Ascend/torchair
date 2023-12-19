from torch._inductor.virtualized import V


# TODO: autogen for all ascir ops
def abs(x):
    graph = V.kernel.graph
    op = graph.add_op("Abs")
    graph.buffer.writeline(f"{op}.x = {x}")
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
