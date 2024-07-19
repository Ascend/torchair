from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.pixel_shuffle.default)
def conveter_aten_pixel_shuffle_default(
    self: Tensor, upscale_factor: int, meta_outputs: TensorSpec = None
):
    """NB: aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.pixel_shuffle.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pixel_shuffle.out)
def conveter_aten_pixel_shuffle_out(
    self: Tensor, upscale_factor: int, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::pixel_shuffle.out(Tensor self, int upscale_factor, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.pixel_shuffle.out ge_converter is not implemented!")
