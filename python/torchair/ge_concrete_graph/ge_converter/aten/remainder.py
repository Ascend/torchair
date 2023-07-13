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


@register_fx_node_ge_converter(torch.ops.aten.remainder.Tensor)
def conveter_aten_remainder_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.remainder.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.Scalar)
def conveter_aten_remainder_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.remainder.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.Scalar_Tensor)
def conveter_aten_remainder_Scalar_Tensor(
        self: Union[Number, Tensor],
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.remainder.Scalar_Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.Tensor_out)
def conveter_aten_remainder_Tensor_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.remainder.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.Scalar_out)
def conveter_aten_remainder_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.remainder.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.Scalar_Tensor_out)
def conveter_aten_remainder_Scalar_Tensor_out(
        self: Union[Number, Tensor],
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.remainder.Scalar_Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.int)
def conveter_aten_remainder_int(
        a: int,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder.int(int a, int b) -> int """
    raise NotImplementedError("torch.ops.aten.remainder.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.float)
def conveter_aten_remainder_float(
        a: float,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder.float(float a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.remainder.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.int_float)
def conveter_aten_remainder_int_float(
        a: int,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder.int_float(int a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.remainder.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.float_int)
def conveter_aten_remainder_float_int(
        a: float,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder.float_int(float a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.remainder.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.remainder.default)
def conveter_aten_remainder_default(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::remainder(Scalar a, Scalar b) -> Scalar """
    raise NotImplementedError("torch.ops.aten.remainder.default ge converter is not implement!")


