import torch
from torch.fx.node import has_side_effect
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import get_default_ge_graph
from ._lib import lib

lib.define(
    """
    scope_exit() -> None
    """
)
has_side_effect(torch.ops.air.scope_exit.default)


@torch.library.impl(lib, "scope_exit", "Meta")
def kernel_meta():
    pass


def kernel_impl():
    raise NotImplementedError("torch.ops.air.scope_exit kernel_impl is not implemented!")


torch.library.impl(lib, "scope_exit", "CPU")(kernel_impl)
torch.library.impl(lib, "scope_exit", "PrivateUse1")(kernel_impl)


def _npu_scope_exit():
    return torch.ops.air.scope_exit()


@register_fx_node_ge_converter(torch.ops.air.scope_exit.default)
def convert_scope_exit():
    graph = get_default_ge_graph()
    graph.pop_attributes()