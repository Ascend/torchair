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


@register_fx_node_ge_converter(torch.ops.aten.fft_ihfft.default)
def conveter_aten_fft_ihfft_default(
    self: Tensor,
    n: Optional[Union[int, Tensor]] = None,
    dim: int = -1,
    norm: Optional[str] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::fft_ihfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.fft_ihfft.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.fft_ihfft.out)
def conveter_aten_fft_ihfft_out(
    self: Tensor,
    n: Optional[Union[int, Tensor]] = None,
    dim: int = -1,
    norm: Optional[str] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::fft_ihfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fft_ihfft.out ge_converter is not implemented!")