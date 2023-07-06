import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
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


@register_fx_node_ge_converter(torch.ops.aten.fft_rfft2.default)
def conveter_aten_fft_rfft2_default(
        self: Tensor,
        s: Optional[Union[List[int], Tensor]] = None,
        dim: List[int] = [-2, -1],
        norm: Optional[str] = None,
        meta_outputs: Any = None):
    """ NB: aten::fft_rfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.fft_rfft2.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fft_rfft2.out)
def conveter_aten_fft_rfft2_out(
        self: Tensor,
        s: Optional[Union[List[int], Tensor]] = None,
        dim: List[int] = [-2, -1],
        norm: Optional[str] = None,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::fft_rfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.fft_rfft2.out ge converter is not implement!")


