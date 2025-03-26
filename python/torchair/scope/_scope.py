import threading
from typing import Any, Dict, List, Tuple
import torch
from torch.fx.node import has_side_effect, Argument, Target
from torchair.ge._ge_graph import compat_as_bytes, get_default_ge_graph
from torchair.core.utils import logger
from ._lib import lib


local = threading.local()


lib.define(
    """
    scope_enter(str[] keys, str[] values) -> None
    """
)
has_side_effect(torch.ops.air.scope_enter.default)


@torch.library.impl(lib, "scope_enter", "Meta")
def kernel_meta(keys: List[str], values: List[str]):
    scope_enter(keys, values)


def kernel_impl(keys: List[str], values: List[str]):
    scope_enter(keys, values)


torch.library.impl(lib, "scope_enter", "CPU")(kernel_impl)
torch.library.impl(lib, "scope_enter", "PrivateUse1")(kernel_impl)


def _npu_scope_enter(attrs):
    keys, values = zip(*attrs)
    keys, values = list(keys), list(values)
    return torch.ops.air.scope_enter(keys, values)


lib.define(
    """
    scope_exit() -> None
    """
)
has_side_effect(torch.ops.air.scope_exit.default)


@torch.library.impl(lib, "scope_exit", "Meta")
def kernel_meta():
    scope_exit()


def kernel_impl():
    scope_exit()


torch.library.impl(lib, "scope_exit", "CPU")(kernel_impl)
torch.library.impl(lib, "scope_exit", "PrivateUse1")(kernel_impl)


def _npu_scope_exit():
    return torch.ops.air.scope_exit()


class ScopeAttrs:
    def __init__(self):
        self._attribute_stack: List[Dict[str, str]] = []

    def push(self, attributes: Dict[str, str]):
        self._attribute_stack.append(attributes)

    def pop(self):
        if self._attribute_stack:
            self._attribute_stack.pop()

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