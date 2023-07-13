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


@register_fx_node_ge_converter(torch.ops.aten.subtract.Tensor)
def conveter_aten_subtract_Tensor(
        self: Tensor,
        other: Tensor,
        *,
        alpha: Union[Number, Tensor] = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.subtract.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.subtract.out)
def conveter_aten_subtract_out(
        self: Tensor,
        other: Tensor,
        *,
        alpha: Union[Number, Tensor] = 1,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::subtract.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.subtract.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.subtract.Scalar)
def conveter_aten_subtract_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        alpha: Union[Number, Tensor] = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.subtract.Scalar ge converter is not implement!")


