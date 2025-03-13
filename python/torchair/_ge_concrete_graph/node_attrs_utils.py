from typing import List
from torchair.core.utils import logger
from torchair.ge._ge_graph import compat_as_bytes, get_default_ge_graph
from . import ge_apis as ge


def set_node_attrs(ge_outputs, *args):
    if ge_outputs is None:
        return
    graph = get_default_ge_graph()
    current_attributes = graph.get_current_attributes()
    if not current_attributes:
        return
    visited = set()
    if isinstance(ge_outputs, ge.Tensor):
        traverse_input(graph, ge_outputs.node, visited, set_node_attributes, *args)
    else:
        for ge_output in ge_outputs:
            traverse_input(graph, ge_output.node, visited, set_node_attributes, *args)


def set_node_attributes(node):
    graph = get_default_ge_graph()
    attributes = graph.get_current_attributes()
    for attr in attributes:
        for key, value in attr.items():
            if key is not None:
                node.attr[key].s = compat_as_bytes(str(value))
                logger.debug(f"Set attribute {key}: {value} on node: {node.name}")


def get_op_from_graph(graph, input_name):
    for op in graph.op:
        if op.name == input_name:
            return op
    raise ValueError(f"can not find {input_name} op in graph {graph.name}")


def is_finished(find_op, *args):
    for arg in args:
        if not isinstance(arg, (list, tuple)):
            if isinstance(arg, ge.Tensor) and (arg.node.name == find_op.name):
                return True
            continue

        for item in arg:
            if isinstance(item, ge.Tensor) and (item.node.name == find_op.name):
                return True
    return find_op.type in {'Data', 'Const'}


def traverse_input(graph, find_op, visited, set_attr_func, *args):
    if is_finished(find_op, *args):
        return
    
    set_attr_func(find_op)
    for input_item in find_op.input:
        if input_item == '':
            continue
        input_name = input_item.split(":")[0]
        if input_name in visited:
            continue
        visited.add(input_name)
        input_op = get_op_from_graph(graph, input_name)
        traverse_input(graph, input_op, visited, set_attr_func, *args)