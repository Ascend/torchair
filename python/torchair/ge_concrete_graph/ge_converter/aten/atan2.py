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


@register_fx_node_ge_converter(torch.ops.aten.atan2.default)
def conveter_aten_atan2_default(
        self: Tensor,
        other: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::atan2(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.atan2.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.out)
def conveter_aten_atan2_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.atan2.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.int)
def conveter_aten_atan2_int(
        a: int,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::atan2.int(int a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.atan2.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.float)
def conveter_aten_atan2_float(
        a: float,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::atan2.float(float a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.atan2.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.int_float)
def conveter_aten_atan2_int_float(
        a: int,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::atan2.int_float(int a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.atan2.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.float_int)
def conveter_aten_atan2_float_int(
        a: float,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::atan2.float_int(float a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.atan2.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.Scalar_Scalar)
def conveter_aten_atan2_Scalar_Scalar(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::atan2.Scalar_Scalar(Scalar a, Scalar b) -> float """
    raise NotImplementedError("torch.ops.aten.atan2.Scalar_Scalar ge converter is not implement!")


