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


@register_fx_node_ge_converter(torch.ops.aten.true_divide.Tensor)
def conveter_aten_true_divide_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.true_divide.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.true_divide.Scalar)
def conveter_aten_true_divide_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.true_divide.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.true_divide.out)
def conveter_aten_true_divide_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.true_divide.out ge converter is not implement!")


