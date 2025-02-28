from typing import List
import torch
from torch.fx.node import has_side_effect
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.node_attrs_utils import push_attributes
from ._lib import lib

lib.define(
    """
    scope_enter(str[] keys, str[] values) -> None
    """
)
has_side_effect(torch.ops.air.scope_enter.default)


@torch.library.impl(lib, "scope_enter", "Meta")
def kernel_meta(keys: List[str], values: List[str]):
    pass


def kernel_impl(keys: List[str], values: List[str]):
    raise NotImplementedError("torch.ops.air.scope_enter kernel_impl is not implemented!")


torch.library.impl(lib, "scope_enter", "CPU")(kernel_impl)
torch.library.impl(lib, "scope_enter", "PrivateUse1")(kernel_impl)


def _npu_scope_enter(keys: List[str], values: List[str]):
    return torch.ops.air.scope_enter(keys, values)


@register_fx_node_ge_converter(torch.ops.air.scope_enter.default)
def convert_scope_enter(keys: List[str], values: List[str]):
    push_attributes(keys, values)