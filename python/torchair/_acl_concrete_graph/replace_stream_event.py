import torch
import torchair
from torchair.ops import (npu_create_tagged_event, npu_tagged_event_record, 
                          npu_tagged_event_wait, npu_record_tagged_stream)
from torchair.core.utils import logger


def replace_stream_event_pass(gm: torch.fx.GraphModule):
    if not hasattr(torch, "npu") or not hasattr(torch.npu, "fake_record_stream"):
        logger.warning("torch has no attr npu or npu.fake_record_stream, skip replace_stream_event_pass")
        return gm
    
    logger.debug(f'after dynamo out, graph is: {gm.graph}')
    stream_stack = []
    new_stream_list = []
    current_stream_map = {}
    current_stream = "default_stream"

    graph = gm.graph
    for node in graph.nodes:
        if node.op == "call_function" and (node.target == torch.npu.Event
                                           or node.target == torch._dynamo.utils.get_user_object_from_id):
            torchair.ops.npu_create_tagged_event(node.name)
        if node.op == "call_method" and node.target == "record":
            node.target = torch.ops.air.tagged_event_record
            node.op = "call_function"
            node.args = (node.args[0].name,)
        if node.op == "call_method" and node.target == "wait":
            node.target = torch.ops.air.tagged_event_wait
            node.op = "call_function"
            node.args = (node.args[0].name,)
        if node.op == "call_function" and node.target == torch.npu.fake_record_stream:
            node.target = torch.ops.air.record_tagged_stream_
            node.args = (node.args[0], node.args[1].name,)

        if node.op == "call_function" and node.target == torch.npu.Stream:
            new_stream_list.append(node)
        if node.op == "call_function" and node.target == torch.npu.current_stream:
            # 将current_stream对应的实际的流记录到字典里
            current_stream_map[node] = current_stream

        # 将set_stream和current_stream转换为入栈和出栈逻辑来创建对应的scope_enter、scope_exit节点
        # 因为current_stream获取的流对象实际上可能和set_stream的对象相同，但是在fx图上是不同的节点
        # 所以函数内部维护了一个current_stream值，每次set_stream时，更新该值为set_stream的传入的流，
        # 下次调用current_stream时，实际获取的就是这个set_stream的传入的流。
        # 通过该方式后续能统一把current_stream当作set_stream来处理
        if node.op == "call_function" and node.target == torch.npu.set_stream:
            # 将current_stream设置为set_stream的传入的流
            current_stream = node
            _to_set_stream = node.args[0]
            if node.args[0] in current_stream_map:
                # 如果传入的是通过current_stream创建的流，从字典里获取它对应的是哪个set_stream创建的流
                _to_set_stream = current_stream_map[node.args[0]]

            if _to_set_stream not in stream_stack and _to_set_stream in new_stream_list:
                # 如果是set_stream创建的流，且不在栈里，则入栈，并创建scope_enter节点
                logger.debug(f"replace_stream_event_pass: push stream: {node.name} to stack")
                stream_stack.append(node)
                node.target = torch.ops.air.scope_enter
                node.args = (["_user_stream_label", "_user_stream_priority"], [_to_set_stream.name, "0"])
            elif _to_set_stream in stream_stack:
                # 如果是已经栈里，则出栈，并创建scope_exit节点
                while(_to_set_stream != stream_stack[-1]):
                    pop_node = stream_stack.pop()
                    logger.debug(f"replace_stream_event_pass: pop stream: {pop_node.name} of stack")
                    with graph.inserting_after(node):
                        graph.call_function(torch.ops.air.scope_exit, args=())
            elif _to_set_stream == "default_stream":
                # 如果是切到初始流，则把栈里元素全部出栈
                while(stream_stack):
                    pop_node = stream_stack.pop()
                    logger.debug(f"replace_stream_event_pass: pop stream: {pop_node.name} of stack")
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