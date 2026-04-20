import torch
import torchair
from torchair.core.utils import logger
from torchair._acl_concrete_graph.graph_pass import _create_event_by_name

# 标记常量，用于标识需要动态解析的 default stream 参数
DEFAULT_STREAM_ID_MARKER = "DEFAULT_STREAM_ID_MARKER"
DEFAULT_DEVICE_INDEX_MARKER = "DEFAULT_DEVICE_INDEX_MARKER"
DEFAULT_DEVICE_TYPE_MARKER = "DEFAULT_DEVICE_TYPE_MARKER"


class GraphCounter:

    COUNT = -1

    @classmethod
    def graph_id(cls):
        cls.COUNT += 1
        return "graph_" + str(cls.COUNT) + "_"
    
    @classmethod
    def set_graph_id(cls, value):
        cls.COUNT = value


def replace_stream_event_pass(gm: torch.fx.GraphModule):
    if not hasattr(torch, "npu") or not hasattr(torch.npu, "fake_record_stream"):
        logger.warning("torch has no attr npu or npu.fake_record_stream, skip replace_stream_event_pass")
        return gm
    
    from torchair.ops._tagged_event import (_npu_create_tagged_event, _npu_tagged_event_record, 
                          _npu_tagged_event_wait, _npu_record_tagged_stream)
    
    logger.debug(f'after dynamo out, graph is: {gm.graph}')
    event_index_to_node_name = {}
    stream_stack = []
    new_stream_list = []
    stream_info_map = {}  # stream_node -> stream_info dict
    current_stream_map = {}
    current_stream = "default_stream"
    default_stream_id = torch.npu.current_stream().stream_id

    graph_id = GraphCounter.graph_id()
    logger.debug(f"replace_stream_event: graph_id is: {graph_id}")
    graph = gm.graph
    is_high_version = _torch_high_version()

    for node in graph.nodes:
        if node.op == "call_function":
            if _is_event(node):
                _create_event_by_name(graph_id + node.name)
                if is_high_version:
                    event_index_to_node_name[node.args[0]] = node.name
            elif is_high_version and node.target == torch.ops.streams.record_event:
                if len(node.args) > 1 and node.args[1] is not None:
                    stream_obj = torch._dynamo.graph_bytecode_inputs.get_external_object_by_index(node.args[1])
                    node.target = torch.ops.air.tagged_event_record_on_stream
                    node.args = (graph_id + event_index_to_node_name[node.args[0]],
                                    stream_obj.stream_id, stream_obj.device_index, stream_obj.device_type, True)
                else:
                    node.target = torch.ops.air.tagged_event_record
                    node.args = (graph_id + event_index_to_node_name[node.args[0]], True)
            elif is_high_version and node.target == torch.ops.streams.wait_event:
                if len(node.args) > 1 and node.args[1] is not None:
                    stream_obj = torch._dynamo.graph_bytecode_inputs.get_external_object_by_index(node.args[1])
                    node.target = torch.ops.air.tagged_event_wait_on_stream
                    node.args = (graph_id + event_index_to_node_name[node.args[0]],
                                    stream_obj.stream_id, stream_obj.device_index, stream_obj.device_type, True)
                else:
                    node.target = torch.ops.air.tagged_event_wait
                    node.args = (graph_id + event_index_to_node_name[node.args[0]], True)    
            elif node.target == torch.npu.fake_record_stream:
                node.target = torch.ops.air.record_tagged_stream_
                node.args = (node.args[0], node.args[1].name,)
            elif _is_stream(node):
                new_stream_list.append(node)
                # Get stream info and save to map
                stream_info_map[node] = _get_stream_info(node)
            elif node.target == torch.npu.current_stream:
                stream_info_map[node] = _get_stream_info(node)
                # 将current_stream对应的实际的流记录到字典里
                current_stream_map[node] = current_stream

        elif node.op == "call_method":
            if node.target == "record":
                # Check if a stream is specified: event.record(stream)
                if len(node.args) > 1:
                    stream_node = node.args[1]  # args[0] = self (event)
                    stream_id, device_index, device_type = _get_stream_info_from_node(stream_node, stream_info_map, default_stream_id)
                    node.target = torch.ops.air.tagged_event_record_on_stream
                    node.op = "call_function"
                    node.args = (graph_id + node.args[0].name, stream_id, device_index, device_type, True)
                else:
                    node.target = torch.ops.air.tagged_event_record
                    node.op = "call_function"
                    node.args = (graph_id + node.args[0].name, True)
            elif node.target == "wait":
                # Check if a stream is specified: event.wait(stream)
                if len(node.args) > 1:
                    stream_node = node.args[1]  # args[0] = self (event)
                    stream_id, device_index, device_type = _get_stream_info_from_node(stream_node, stream_info_map, default_stream_id)
                    node.target = torch.ops.air.tagged_event_wait_on_stream
                    node.op = "call_function"
                    node.args = (graph_id + node.args[0].name, stream_id, device_index, device_type, True)
                else:
                    node.target = torch.ops.air.tagged_event_wait
                    node.op = "call_function"
                    node.args = (graph_id + node.args[0].name, True)
            elif node.target == "wait_event":
                # stream.wait_event(event) - reuse tagged_event_wait_on_stream
                stream_node = node.args[0]  # self (stream)
                event_node = node.args[1]   # event
                stream_id, device_index, device_type = _get_stream_info_from_node(stream_node, stream_info_map, default_stream_id)
                node.target = torch.ops.air.tagged_event_wait_on_stream
                node.op = "call_function"
                node.args = (graph_id + event_node.name, stream_id, device_index, device_type, True)
            elif node.target == "wait_stream":
                stream_node = node.args[0]       # self (stream)
                other_stream_node = node.args[1] # other_stream
                stream_id, device_index, device_type = _get_stream_info_from_node(stream_node, stream_info_map, default_stream_id)
                other_stream_id, other_device_index, other_device_type = _get_stream_info_from_node(other_stream_node, stream_info_map, default_stream_id)
                node.target = torch.ops.air.tagged_stream_wait_stream
                node.op = "call_function"
                node.args = (stream_id, device_index, device_type, other_stream_id, other_device_index, other_device_type)
            elif node.target == "record_event":
                # stream.record_event(event) - reuse tagged_event_record_on_stream
                stream_node = node.args[0]  # self (stream)
                stream_id, device_index, device_type = _get_stream_info_from_node(stream_node, stream_info_map, default_stream_id)
                if len(node.args) > 1 and node.args[1] is not None:
                    event_node = node.args[1]  # event
                    node.target = torch.ops.air.tagged_event_record_on_stream
                    node.op = "call_function"
                    node.args = (graph_id + event_node.name, stream_id, device_index, device_type, True)
                    node.replace_all_uses_with(event_node)
                else:
                    # event=None: create new event and register, then reuse tagged_event_record_on_stream
                    new_event_tag = graph_id + node.name
                    _create_event_by_name(new_event_tag)
                    node.target = torch.ops.air.tagged_event_record_on_stream
                    node.op = "call_function"
                    node.args = (new_event_tag, stream_id, device_index, device_type, True)

        # 将set_stream和current_stream转换为入栈和出栈逻辑来创建对应的scope_enter、scope_exit节点
        # 因为current_stream获取的流对象实际上可能和set_stream的对象相同，但是在fx图上是不同的节点
        # 所以函数内部维护了一个current_stream值，每次set_stream时，更新该值为set_stream的传入的流，
        # 下次调用current_stream时，实际获取的就是这个set_stream的传入的流。
        # 通过该方式后续能统一把current_stream当作set_stream来处理
        if node.op == "call_function" and node.target == torch.npu.set_stream:
            _to_set_stream = node.args[0]
            if node.args[0] in current_stream_map:
                # 如果传入的是通过current_stream创建的流，从字典里获取它对应的是哪个set_stream创建的流
                _to_set_stream = current_stream_map[node.args[0]]

            # 将current_stream设置为set_stream的传入的流
            current_stream = _to_set_stream  

            if _to_set_stream not in stream_stack and _to_set_stream in new_stream_list:
                # 如果是set_stream创建的流，且不在栈里，则入栈，并创建scope_enter节点
                logger.debug(f"replace_stream_event: push stream: {graph_id + node.name} to stack")
                stream_stack.append(_to_set_stream)
                node.target = torch.ops.air.scope_enter
                node.args = _build_scope_enter_args(stream_info_map, _to_set_stream, graph_id, default_stream_id)
            elif _to_set_stream in stream_stack:
                # 如果是已经栈里，则出栈，并创建scope_exit节点
                while _to_set_stream != stream_stack[-1]:
                    pop_node = stream_stack.pop()
                    logger.debug(f"replace_stream_event: pop stream: {graph_id + pop_node.name} of stack")
                    with graph.inserting_after(node):
                        graph.call_function(torch.ops.air.scope_exit, args=())
            elif _to_set_stream == "default_stream":
                # 如果是切到初始流，则把栈里元素全部出栈
                while stream_stack:
                    pop_node = stream_stack.pop()
                    logger.debug(f"replace_stream_event: pop stream: {graph_id + pop_node.name} of stack")
                    with graph.inserting_after(node):
                        graph.call_function(torch.ops.air.scope_exit, args=())
            else:
                raise RuntimeError(f"Can not find stream: {_to_set_stream.name} in FX graph when parse torch.npu.set_stream."
                                   "Please submit a issue to TorchAir")

    if stream_stack:
        logger.error(f"there is some stream not pop from stack: {stream_stack}")
        raise RuntimeError("When use npugraph_ex, you must make sure at the end of your code set stream to the same stream "
                        "as the begin of your code.\n"
                        "E.g: at the begin of your code, call 'enter_stream = torch.npu.current_stream()', "
                        "and at the end of your code, call 'torch.npu.set_stream(enter_stream)'. \n"
                        "A more recommended way is to use 'with torch.npu.stream()' instead of 'torch.npu.set_stream()'")


    gm.recompile()
    logger.debug(f'after replace_stream_event, graph is: {gm.graph}')
    return gm


def _is_event(node):
    if node.target == torch.npu.Event:
        return True
    if hasattr(torch._dynamo.utils, "get_user_object_from_id") and node.target == torch._dynamo.utils.get_user_object_from_id:
        return True
    if (_torch_high_version()
        and node.target == torch._dynamo.graph_bytecode_inputs.get_external_object_by_index):
        if isinstance(torch._dynamo.graph_bytecode_inputs.get_external_object_by_index(node.args[0]), torch.npu.Event):
            return True
    return False


def _is_stream(node):
    if node.target == torch.npu.Stream:
        return True
    if (_torch_high_version()
        and node.target == torch._dynamo.graph_bytecode_inputs.get_external_object_by_index):
        if isinstance(torch._dynamo.graph_bytecode_inputs.get_external_object_by_index(node.args[0]), torch.npu.Stream):
            return True
    return False


def _get_stream_info(stream_node):
    """
    Get stream information (stream_id, device_index, device_type) from a stream node.

    For both internally created and externally passed streams, extract info from
    stream_node.meta.get('example_value').

    Returns:
        dict: {"stream_id": int or None, "device_index": int or None,
               "device_type": int or None, "priority": int}
    """
    stream_info = {"stream_id": None, "device_index": None, "device_type": None, "priority": 0}

    # Get stream info from stream_node.meta['example_value']
    example_value = stream_node.meta.get('example_value', None)
    if example_value is None:
        raise RuntimeError(f"Can't read stream info from node: {stream_node.name}, "
                           f"cause there is no example_value in stream_node.meta: {stream_node.meta}")

    if (not hasattr(example_value, "stream_id") 
        or not hasattr(example_value, "device_index") 
        or not hasattr(example_value, "device_type")):
        raise RuntimeError(f"Can't read stream info from node: {stream_node.name},"
                           f" cause there is no stream_id/device_index/device_type"
                           f" in stream_node.meta.example_value: {example_value}")

    stream_info["stream_id"] = example_value.stream_id
    stream_info["device_index"] = example_value.device_index
    stream_info["device_type"] = example_value.device_type

    logger.debug(f"_get_stream_info: stream_node={stream_node.name}, info={stream_info}")
    return stream_info


def _get_stream_info_from_node(stream_node, stream_info_map, default_stream_id=None):
    """
    Extract stream info (stream_id, device_index, device_type) from stream_info_map.

    Args:
        stream_node: The FX node representing a stream
        stream_info_map: Dict mapping stream_node to stream info
        default_stream_id: The default stream id captured at the beginning of replace_stream_event_pass.
                          If stream_id equals default_stream_id, markers will be returned instead.

    Returns:
        tuple: (stream_id, device_index, device_type)
    """
    stream_info = stream_info_map.get(stream_node, {})
    if not stream_info:
        raise RuntimeError(f"Can't get stream info from node:{stream_node.name}, stream_info is: {stream_info_map}")
    stream_id = stream_info.get("stream_id")
    device_index = stream_info.get("device_index")
    device_type = stream_info.get("device_type")

    # 如果 stream_id 等于 default_stream_id，则替换为标记字符串
    if default_stream_id is not None and stream_id == default_stream_id:
        stream_id = DEFAULT_STREAM_ID_MARKER
        device_index = DEFAULT_DEVICE_INDEX_MARKER
        device_type = DEFAULT_DEVICE_TYPE_MARKER

    return str(stream_id), str(device_index), str(device_type)


def _build_scope_enter_args(stream_info_map: dict, stream_node, graph_id: str, default_stream_id=None):
    """
    Build scope_enter args with stream info.

    Args:
        stream_info_map: Dict mapping stream_node to stream info
        stream_node: The stream node to build args for
        graph_id: The graph ID prefix for the stream label
        default_stream_id: The default stream id captured at the beginning of replace_stream_event_pass.
                          If stream_id equals default_stream_id, markers will be used instead.

    Returns:
        tuple: (keys, values) where keys and values are lists for scope_enter args
    """
    # Get stream info from stream_info_map
    stream_info = stream_info_map.get(stream_node, {})
    stream_id = stream_info.get("stream_id", None)
    device_index = stream_info.get("device_index", None)
    device_type = stream_info.get("device_type", None)
    priority = 0  # Fixed priority

    # 如果 stream_id 等于 default_stream_id，则替换为标记字符串
    if default_stream_id is not None and stream_id == default_stream_id:
        stream_id = DEFAULT_STREAM_ID_MARKER
        device_index = DEFAULT_DEVICE_INDEX_MARKER
        device_type = DEFAULT_DEVICE_TYPE_MARKER

    # Build scope_enter args with stream info
    # Format: (keys, values) where keys and values are lists
    keys = ["_user_stream_label", "_user_stream_priority"]
    values = [graph_id + stream_node.name, str(priority)]

    # Add stream_id, device_index, device_type if available
    if stream_id is not None:
        keys.append("_user_stream_id")
        # 标记字符串直接使用，非标记值需要 str() 转换
        values.append(stream_id if stream_id == DEFAULT_STREAM_ID_MARKER else str(stream_id))
    if device_index is not None:
        keys.append("_user_stream_device_index")
        values.append(device_index if device_index == DEFAULT_DEVICE_INDEX_MARKER else str(device_index))
    if device_type is not None:
        keys.append("_user_stream_device_type")
        values.append(device_type if device_type == DEFAULT_DEVICE_TYPE_MARKER else str(device_type))

    return keys, values


def _torch_high_version():
    if (hasattr(torch._dynamo, "graph_bytecode_inputs")
        and hasattr(torch._dynamo.graph_bytecode_inputs, "get_external_object_by_index")):
        return True
    return False
