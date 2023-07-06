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


@register_fx_node_ge_converter(torch.ops.aten.fft_rfft.default)
def conveter_aten_fft_rfft_default(
        self: Tensor,
        n: Optional[Union[int, Tensor]] = None,
        dim: int = -1,
        norm: Optional[str] = None,
        meta_outputs: Any = None):
    """ NB: aten::fft_rfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.fft_rfft.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fft_rfft.out)
def conveter_aten_fft_rfft_out(
        self: Tensor,
        n: Optional[Union[int, Tensor]] = None,
        dim: int = -1,
        norm: Optional[str] = None,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::fft_rfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.fft_rfft.out ge converter is not implement!")


