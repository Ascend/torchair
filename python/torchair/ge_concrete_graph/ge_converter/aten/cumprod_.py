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


@register_fx_node_ge_converter(torch.ops.aten.cumprod_.default)
def conveter_aten_cumprod__default(
        self: Tensor,
        dim: int,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Any = None):
    """ NB: aten::cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.cumprod_.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod_.dimname)
def conveter_aten_cumprod__dimname(
        self: Tensor,
        dim: str,
        *,
        dtype: Optional[int] = None,
        meta_outputs: Any = None):
    """ NB: aten::cumprod_.dimname(Tensor(a!) self, str dim, *, ScalarType? dtype=None) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.cumprod_.dimname ge converter is not implement!")


