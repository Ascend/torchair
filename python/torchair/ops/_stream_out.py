import torch
from torch.fx.node import has_side_effect
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.stream_utils import set_stream_tag, set_stream_priority
from ._lib import lib

lib.define(
    """
    stream_out(str stream_tag, int stream_priority) -> None
    """
)
has_side_effect(torch.ops.air.stream_out.default)


@torch.library.impl(lib, "stream_out", "Meta")
def kernel_meta(stream_tag: str, stream_priority: int = 0):
    pass


def kernel_impl(stream_tag: str, stream_priority: int = 0):
    raise NotImplementedError("torch.ops.air.stream_out kernel_impl is not implemented!")


torch.library.impl(lib, "stream_out", "CPU")(kernel_impl)
torch.library.impl(lib, "stream_out", "PrivateUse1")(kernel_impl)


def _npu_stream_out(stream_tag: str, stream_priority: int = 0):
    return torch.ops.air.stream_out(stream_tag, stream_priority)


@register_fx_node_ge_converter(torch.ops.air.stream_out.default)
def convert_stream_out(stream_tag: str, stream_priority: int = 0):
    set_stream_tag(None)
    set_stream_priority(0)