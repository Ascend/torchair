import torch
from torch.fx.node import has_side_effect
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.scope_options_utils import set_scope, set_options
from ._lib import lib

lib.define(
    """
    kernel_scope_in(str scope, str options) -> None
    """
)
has_side_effect(torch.ops.air.kernel_scope_in.default)


@torch.library.impl(lib, "kernel_scope_in", "Meta")
def kernel_meta(scope: str, options: str = None):
    pass


def kernel_impl(scope: str, options: str = None):
    raise NotImplementedError("torch.ops.air.kernel_scope_in kernel_impl is not implemented!")


torch.library.impl(lib, "kernel_scope_in", "CPU")(kernel_impl)
torch.library.impl(lib, "kernel_scope_in", "PrivateUse1")(kernel_impl)


def _npu_kernel_scope_in(scope: str, options: str = None):
    return torch.ops.air.kernel_scope_in(scope, options)


@register_fx_node_ge_converter(torch.ops.air.kernel_scope_in.default)
def convert_kernel_scope_in(scope: str, options: str = None):
    set_scope(scope)
    set_options(options)