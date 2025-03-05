from torchair.core.utils import logger
from torchair.ge._ge_graph import compat_as_bytes, get_default_ge_graph
from . import ge_apis as ge


def set_stream_info(ge_outputs, *args):
    stream_tag = get_default_ge_graph().stream_tag
    if (stream_tag is None) or (ge_outputs is None):
        return
    
    visited = set()
    graph = get_default_ge_graph()
    if isinstance(ge_outputs, ge.Tensor):
        traverse_input(graph, ge_outputs.node, visited, *args)
    else:
        for ge_output in ge_outputs:
            traverse_input(graph, ge_output.node, visited, *args)


def set_stream_label_and_priority(node):
    graph = get_default_ge_graph()
    stream_tag = graph.stream_tag
    stream_priority = graph.stream_priority
    logger.debug(f"set stream_label: {stream_tag}, stream_priority: {stream_priority}, op: {node.name}")
    node.attr["_user_stream_label"].s = compat_as_bytes(stream_tag)
    node.attr["_user_stream_priority"].s = compat_as_bytes(str(stream_priority))


def get_op_from_graph(graph, input_name):
    for op in graph.op:
        if op.name == input_name:
            return op
    raise ValueError(f"can not find {input_name} op in graph {graph.name}")


def is_finished(find_op, *args):
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, ge.Tensor) and (item.node.name == find_op.name):
                    return True
            continue
        if isinstance(arg, ge.Tensor) and (arg.node.name == find_op.name):
            return True
    return find_op.type in {'Data', 'Const'}


def traverse_input(graph, find_op, visited, *args):
    if is_finished(find_op, *args):
        return
    
    set_stream_label_and_priority(find_op)
    for input_item in find_op.input:
        if input_item == '':
            continue
        input_name = input_item.split(":")[0]
        if input_name in visited:
            continue
        visited.add(input_name)
        find_op = get_op_from_graph(graph, input_name)
        traverse_input(graph, find_op, visited, *args)
            

def set_stream_tag(stream_tag: str):
    logger.debug(f"set graph stream tag: {stream_tag}")
    graph = get_default_ge_graph()
    graph.set_stream_tag(stream_tag)


def set_stream_priority(stream_priority: int):
    logger.debug(f"set graph stream priority: {stream_priority}")
    graph = get_default_ge_graph()
    graph.set_stream_priority(stream_priority)

