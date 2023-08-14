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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.fft_fft2.default)
def conveter_aten_fft_fft2_default(
    self: Tensor,
    s: Optional[Union[List[int], Tensor]] = None,
    dim: List[int] = (),
    norm: Optional[str] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::fft_fft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.fft_fft2.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.fft_fft2.out)
def conveter_aten_fft_fft2_out(
    self: Tensor,
    s: Optional[Union[List[int], Tensor]] = None,
    dim: List[int] = (),
    norm: Optional[str] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::fft_fft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fft_fft2.out ge_converter is not implemented!")
