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


@register_fx_node_ge_converter(torch.ops.aten.fft_ifftn.default)
def conveter_aten_fft_ifftn_default(
        self: Tensor,
        s: Optional[Union[List[int], Tensor]] = None,
        dim: Optional[List[int]] = None,
        norm: Optional[str] = None,
        meta_outputs: Any = None):
    """ NB: aten::fft_ifftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.fft_ifftn.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fft_ifftn.out)
def conveter_aten_fft_ifftn_out(
        self: Tensor,
        s: Optional[Union[List[int], Tensor]] = None,
        dim: Optional[List[int]] = None,
        norm: Optional[str] = None,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::fft_ifftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.fft_ifftn.out ge converter is not implement!")


