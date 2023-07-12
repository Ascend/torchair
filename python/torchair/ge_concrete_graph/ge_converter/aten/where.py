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


@register_fx_node_ge_converter(torch.ops.aten.where.self)
def conveter_aten_where_self(
        condition: Tensor,
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.where.self ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.where.ScalarOther)
def conveter_aten_where_ScalarOther(
        condition: Tensor,
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.where.ScalarOther ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.where.ScalarSelf)
def conveter_aten_where_ScalarSelf(
        condition: Tensor,
        self: Union[Number, Tensor],
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.where.ScalarSelf ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.where.Scalar)
def conveter_aten_where_Scalar(
        condition: Tensor,
        self: Union[Number, Tensor],
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.where.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.where.default)
def conveter_aten_where_default(
        condition: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::where(Tensor condition) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten.where.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.where.self_out)
def conveter_aten_where_self_out(
        condition: Tensor,
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::where.self_out(Tensor condition, Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.where.self_out ge converter is not implement!")


