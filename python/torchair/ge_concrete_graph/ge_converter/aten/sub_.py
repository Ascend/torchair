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


@register_fx_node_ge_converter(torch.ops.aten.sub_.Tensor)
def conveter_aten_sub__Tensor(
        self: Tensor,
        other: Tensor,
        *,
        alpha: Union[Number, Tensor] = 1,
        meta_outputs: Any = None):
    """ NB: aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.sub_.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub_.Scalar)
def conveter_aten_sub__Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        alpha: Union[Number, Tensor] = 1,
        meta_outputs: Any = None):
    """ NB: aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.sub_.Scalar ge converter is not implement!")


