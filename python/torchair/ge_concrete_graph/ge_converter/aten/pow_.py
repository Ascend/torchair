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


@register_fx_node_ge_converter(torch.ops.aten.pow_.Scalar)
def conveter_aten_pow__Scalar(
        self: Tensor,
        exponent: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.pow_.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow_.Tensor)
def conveter_aten_pow__Tensor(
        self: Tensor,
        exponent: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.pow_.Tensor ge converter is not implement!")


