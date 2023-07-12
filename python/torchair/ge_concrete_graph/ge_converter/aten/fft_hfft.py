import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten.fft_hfft.default)
def conveter_aten_fft_hfft_default(
        self: Tensor,
        n: Optional[Union[int, Tensor]] = None,
        dim: int = -1,
        norm: Optional[str] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::fft_hfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.fft_hfft.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fft_hfft.out)
def conveter_aten_fft_hfft_out(
        self: Tensor,
        n: Optional[Union[int, Tensor]] = None,
        dim: int = -1,
        norm: Optional[str] = None,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::fft_hfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.fft_hfft.out ge converter is not implement!")


