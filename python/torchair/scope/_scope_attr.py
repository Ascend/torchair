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
        self._attribute_stack.append(attributes)

    def pop(self):
        if self._attribute_stack:
            self._attribute_stack.pop()
            logger.debug(f"ScopeAttrs pop attrs, the lens of stack is: {len(self._attribute_stack)}")

    def apply(self, op):
        for attrs in self._attribute_stack:
            for key, value in attrs.items():
                op.attr[key].s = compat_as_bytes(str(value))
                logger.debug(f"Set attribute {key}: {value} on op: {op.name}")


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
        ge_outputs = func(self, target, args, kwargs, meta_outputs)
        apply_scope_attr(graph.op[num_ops:])
        return ge_outputs

    return wrapper


def has_scope_attr():
    return hasattr(local, 'scope_attr') and len(local.scope_attr._attribute_stack) > 0
    