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


@register_fx_node_ge_converter(torch.ops.aten.copysign.Tensor)
def conveter_aten_copysign_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.copysign.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copysign.Scalar)
def conveter_aten_copysign_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copysign.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.copysign.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copysign.out)
def conveter_aten_copysign_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.copysign.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copysign.Scalar_out)
def conveter_aten_copysign_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copysign.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.copysign.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copysign.int)
def conveter_aten_copysign_int(
        a: int,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copysign.int(int a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.copysign.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copysign.float)
def conveter_aten_copysign_float(
        a: float,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copysign.float(float a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.copysign.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copysign.int_float)
def conveter_aten_copysign_int_float(
        a: int,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copysign.int_float(int a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.copysign.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copysign.float_int)
def conveter_aten_copysign_float_int(
        a: float,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copysign.float_int(float a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.copysign.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copysign.default)
def conveter_aten_copysign_default(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copysign(Scalar a, Scalar b) -> float """
    raise NotImplementedError("torch.ops.aten.copysign.default ge converter is not implement!")


