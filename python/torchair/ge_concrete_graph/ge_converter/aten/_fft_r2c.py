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


@register_fx_node_ge_converter(torch.ops.aten._fft_r2c.default)
def conveter_aten__fft_r2c_default(
        self: Tensor,
        dim: List[int],
        normalization: int,
        onesided: bool,
        meta_outputs: Any = None):
    """ NB: aten::_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor """
    raise NotImplementedError("torch.ops.aten._fft_r2c.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._fft_r2c.out)
def conveter_aten__fft_r2c_out(
        self: Tensor,
        dim: List[int],
        normalization: int,
        onesided: bool,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::_fft_r2c.out(Tensor self, int[] dim, int normalization, bool onesided, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten._fft_r2c.out ge converter is not implement!")


