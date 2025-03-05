from torchair.core.utils import logger
from torchair.ge._ge_graph import compat_as_bytes, get_default_ge_graph
from . import ge_apis as ge


def set_kernel_scope(ge_outputs, *args):
    scope = get_default_ge_graph().scope
    if (scope is None) or (ge_outputs is None):
        return
    
    visited = set()
    graph = get_default_ge_graph()
    if isinstance(ge_outputs, ge.Tensor):
        traverse_input(graph, ge_outputs.node, visited, *args)
    else:
        for ge_output in ge_outputs:
            traverse_input(graph, ge_output.node, visited, *args)


def set_scope_options(node):
    graph = get_default_ge_graph()
    scope = graph.scope
    options = graph.options
    logger.debug(f"set scope: {scope}, options: {options}, op: {node.name}")
    node.attr["_super_kernel_scope"].s = compat_as_bytes(scope)
    node.attr["_super_kernel_options"].s = compat_as_bytes(options)


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
    
    set_scope_options(find_op)
    for input_item in find_op.input:
        if input_item == '':
            continue
        input_name = input_item.split(":")[0]
        if input_name in visited:
            continue
        visited.add(input_name)
        find_op = get_op_from_graph(graph, input_name)
        traverse_input(graph, find_op, visited, *args)


def set_scope(scope: str):
    logger.debug(f"set graph scope: {scope}")
    graph = get_default_ge_graph()
    graph.set_scope(scope)


def set_options(options: str):
    logger.debug(f"set graph options: {options}")
    graph = get_default_ge_graph()
    graph.set_options(options)

