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


@register_fx_node_ge_converter(torch.ops.aten._fft_c2c.default)
def conveter_aten__fft_c2c_default(
        self: Tensor,
        dim: Union[List[int], Tensor],
        normalization: int,
        forward: bool,
        meta_outputs: Any = None):
    """ NB: aten::_fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor """
    raise NotImplementedError("torch.ops.aten._fft_c2c.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._fft_c2c.out)
def conveter_aten__fft_c2c_out(
        self: Tensor,
        dim: Union[List[int], Tensor],
        normalization: int,
        forward: bool,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::_fft_c2c.out(Tensor self, SymInt[] dim, int normalization, bool forward, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten._fft_c2c.out ge converter is not implement!")


