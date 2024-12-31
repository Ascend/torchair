import torch
from torch.fx.node import has_side_effect
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.stream_utils import set_stream_tag, set_stream_priority
from ._lib import lib

lib.define(
    """
    stream_in(str stream_tag, int stream_priority) -> None
    """
)
has_side_effect(torch.ops.air.stream_in.default)


@torch.library.impl(lib, "stream_in", "Meta")
def kernel_meta(stream_tag: str, stream_priority: int = 0):
    pass


def kernel_impl(stream_tag: str, stream_priority: int = 0):
    raise NotImplementedError("torch.ops.air.stream_in kernel_impl is not implemented!")


torch.library.impl(lib, "stream_in", "CPU")(kernel_impl)
torch.library.impl(lib, "stream_in", "PrivateUse1")(kernel_impl)


def _npu_stream_in(stream_tag: str, stream_priority: int = 0):
    return torch.ops.air.stream_in(stream_tag, stream_priority)


@register_fx_node_ge_converter(torch.ops.air.stream_in.default)
def convert_stream_in(stream_tag: str, stream_priority: int = 0):
    set_stream_tag(stream_tag)
    set_stream_priority(stream_priority)