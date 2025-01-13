from torchair.ge._ge_graph import compat_as_bytes, get_default_ge_graph
from . import ge_apis as ge


def set_stream_info(ge_outputs, *args):
    stream_tag = get_default_ge_graph().stream_tag
    if stream_tag is not None:
        visited = set()
        graph = get_default_ge_graph()
        if isinstance(ge_outputs, ge.Tensor):
            set_stream_label_and_priority(ge_outputs.node)
            traverse_input(graph, ge_outputs.node.input, visited, *args)
        elif isinstance(ge_outputs, (list, tuple)) and all([isinstance(v, ge.Tensor) for v in ge_outputs]):
            for ge_output in ge_outputs:
                set_stream_label_and_priority(ge_output.node)
                traverse_input(graph, ge_output.node.input, visited, *args)


def set_stream_label_and_priority(node):
    graph = get_default_ge_graph()
    stream_tag = graph.stream_tag
    stream_priority = graph.stream_priority
    node.attr["_user_stream_label"].s = compat_as_bytes(stream_tag)
    node.attr["_user_stream_priority"].s = compat_as_bytes(str(stream_priority))


def get_op_from_graph(graph, input_name):
    for op in graph.op:
        if op.name == input_name:
            return op
    return None


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


def traverse_input(graph, inputs, visited, *args):
    for index, input_item in enumerate(inputs):
        if input_item == '':
            continue
        input_name_list = input_item.split(":")
        input_name = input_name_list[0]
        if input_name not in visited:
            visited.add(input_name)
            find_op = get_op_from_graph(graph, input_name)
            if find_op is None:
                raise ValueError(f"can not find {input_name} op in graph {graph.name}")
            if not is_finished(find_op, *args):
                set_stream_label_and_priority(find_op)
                traverse_input(graph, find_op.input, visited, *args)


def set_stream_tag(stream_tag: str):
    graph = get_default_ge_graph()
    graph.set_stream_tag(stream_tag)


def set_stream_priority(stream_priority: int):
    graph = get_default_ge_graph()
    graph.set_stream_priority(stream_priority)