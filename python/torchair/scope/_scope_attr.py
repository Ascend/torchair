import threading
from typing import Any, Dict, List, Tuple
import torch
from torch.fx.node import Argument, Target
from torchair.ge._ge_graph import compat_as_bytes, get_default_ge_graph
from torchair.core.utils import logger


local = threading.local()

 
class ScopeAttrs:
    def __init__(self):
        logger.debug(f"ScopeAttrs init")
        self._attribute_stack: List[Dict[str, str]] = []

    def push(self, attributes: Dict[str, str]):
        logger.debug(f"ScopeAttrs push attrs: {attributes}")

        _process_on_push(self._attribute_stack, attributes)

        self._attribute_stack.append(attributes)

    def pop(self):
        if not self._attribute_stack:
            return
        logger.debug(f"ScopeAttrs pop attrs, the lens of stack is: {len(self._attribute_stack)}")
        pop_attribute_stack = self._attribute_stack.pop()

        _process_on_pop(pop_attribute_stack)

    def top(self):
        if self._attribute_stack and len(self._attribute_stack) > 0:
            return self._attribute_stack[-1]
        return None

    def apply(self, op):
        for attrs in self._attribute_stack:
            if "_profiler_trace_index" in attrs:
                if attrs["_first_op"] is None:
                    attrs["_first_op"] = op
                attrs["_last_op"] = op
                continue # 其他算子不需要打profiler属性

            for key, value in attrs.items():
                if key == "_op_exec_never_timeout":
                    op.attr[key].b = (value == "True")
                else:
                    op.attr[key].s = compat_as_bytes(str(value))
                logger.debug(f"Set attribute {key}: {value} on op: {op.name}")

    def scope_attr_info(self):
        return self._attribute_stack                  


def scope_enter(keys: List[str], values: List[str]):
    if not hasattr(local, 'scope_attr'):
        local.scope_attr = ScopeAttrs()
    local.scope_attr.push(dict(zip(keys, values)))


def scope_exit():
    if hasattr(local, 'scope_attr'):
        local.scope_attr.pop()


def apply_scope_attr(ops):
    if hasattr(local, 'scope_attr'):
        for op in ops:
            local.scope_attr.apply(op)


def guard_scope_attr(func):
    def wrapper(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        graph = get_default_ge_graph()
        num_ops = len(graph.op)
        with self.scope_attr_ctx(local.scope_attr.scope_attr_info() if hasattr(local, 'scope_attr') else []):        
            ge_outputs = func(self, target, args, kwargs, meta_outputs)
        apply_scope_attr(graph.op[num_ops:]) 
        return ge_outputs

    return wrapper


def has_scope_attr():
    return hasattr(local, 'scope_attr') and len(local.scope_attr._attribute_stack) > 0


_GLOBAL_TAG_TO_STREAM = {}
_GLOBAL_TAGGED_STREAM_LOCK = threading.Lock()


def _npu_get_or_create_stream_with_tag(tag: str):
    with _GLOBAL_TAGGED_STREAM_LOCK:
        if tag in _GLOBAL_TAG_TO_STREAM.keys():
            logger.debug(f"get stream with tag = {tag}, stream = {_GLOBAL_TAG_TO_STREAM[tag]} successfully")
            return _GLOBAL_TAG_TO_STREAM[tag]
        import torch_npu
        stream = torch_npu.npu.Stream()
        _GLOBAL_TAG_TO_STREAM[tag] = stream
        logger.debug(f"create stream = {stream} with tag = {tag} successfully")
        return stream


def guard_with_user_stream_scope(func):
    def wrapper(self, node):
        if not has_scope_attr() or local.scope_attr.top().get("_user_stream_label", None) is None:
            return func(self, node)
        user_stream_label = local.scope_attr.top().get("_user_stream_label")
        target_stream = _npu_get_or_create_stream_with_tag(user_stream_label)
        logger.debug(f"guard with user stream scope, node = {node}, "
                     f"user stream label = {user_stream_label}, target_stream = {target_stream}")
        with torch.npu.stream(target_stream):
            return func(self, node)
    return wrapper


def _process_on_push(attribute_stack, attributes):
    if "_profiler_trace_index" in attributes:
        for existing in attribute_stack:
            if "_profiler_trace_index" in existing:
                raise ValueError(f"Nested profiler_trace is not allowed")
        attributes["_first_op"] = None
        attributes["_last_op"] = None


def _process_on_pop(pop_attribute_stack):
    if "_profiler_trace_index" not in pop_attribute_stack or pop_attribute_stack["_first_op"] is None:
        return

    idx = pop_attribute_stack["_profiler_trace_index"]
    mode = pop_attribute_stack["_profiler_trace_pos"]
    first_op = pop_attribute_stack["_first_op"]
    last_op = pop_attribute_stack["_last_op"]
    single = (first_op is last_op)

    if mode in ("begin", "both"):
        pos = "both" if (single and mode == "both") else "begin"
        first_op.attr["_profiler_trace_index"].s = compat_as_bytes(idx)
        first_op.attr["_profiler_trace_pos"].s = compat_as_bytes(pos)

    need_end = (mode == "end") or (mode == "both" and not single)
    if need_end:
        target_op = last_op if not single else first_op
        target_op.attr["_profiler_trace_index"].s = compat_as_bytes(idx)
        target_op.attr["_profiler_trace_pos"].s = compat_as_bytes("end")
