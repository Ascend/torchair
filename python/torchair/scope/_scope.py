from typing import List

import torch
from torch.fx.node import has_side_effect
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter

from ._lib import lib
from ._scope_attr import scope_enter, scope_exit


lib.define(
    """
    scope_enter(str[] keys, str[] values, bool need_excute=False) -> None
    """
)
has_side_effect(torch.ops.air.scope_enter.default)


@torch.library.impl(lib, "scope_enter", "Meta")
def kernel_meta(keys: List[str], values: List[str], need_excute=False):
    if need_excute:
        scope_enter(keys, values)


def kernel_impl(keys: List[str], values: List[str], need_excute=False):
    scope_enter(keys, values)


torch.library.impl(lib, "scope_enter", "CPU")(kernel_impl)
torch.library.impl(lib, "scope_enter", "PrivateUse1")(kernel_impl)


def _npu_scope_enter(attrs):
    keys, values = zip(*attrs)
    keys, values = list(keys), list(values)
    return torch.ops.air.scope_enter(keys, values)


lib.define(
    """
    scope_exit(bool need_excute=False) -> None
    """
)
has_side_effect(torch.ops.air.scope_exit.default)


@torch.library.impl(lib, "scope_exit", "Meta")
def kernel_meta(need_excute=False):
    if need_excute:
        scope_exit()


def kernel_impl(need_excute=False):
    scope_exit()


torch.library.impl(lib, "scope_exit", "CPU")(kernel_impl)
torch.library.impl(lib, "scope_exit", "PrivateUse1")(kernel_impl)


def _npu_scope_exit():
    return torch.ops.air.scope_exit()