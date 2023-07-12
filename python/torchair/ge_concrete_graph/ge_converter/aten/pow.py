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


@register_fx_node_ge_converter(torch.ops.aten.pow.Tensor_Tensor)
def conveter_aten_pow_Tensor_Tensor(
        self: Tensor,
        exponent: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor """
    raise NotImplementedError("torch.ops.aten.pow.Tensor_Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.Tensor_Scalar)
def conveter_aten_pow_Tensor_Scalar(
        self: Tensor,
        exponent: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor """
    # Notes: Castlike is a solution to not add dtype attributes during the concrete graph
    cast_like = ge.CastLike(exponent, self)
    return ge.Pow(self, cast_like)


@register_fx_node_ge_converter(torch.ops.aten.pow.Scalar)
def conveter_aten_pow_Scalar(
        self: Union[Number, Tensor],
        exponent: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor """
    raise NotImplementedError("torch.ops.aten.pow.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.Scalar_out)
def conveter_aten_pow_Scalar_out(
        self: Union[Number, Tensor],
        exponent: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.pow.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.Tensor_Scalar_out)
def conveter_aten_pow_Tensor_Scalar_out(
        self: Tensor,
        exponent: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.pow.Tensor_Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.Tensor_Tensor_out)
def conveter_aten_pow_Tensor_Tensor_out(
        self: Tensor,
        exponent: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.pow.Tensor_Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.int)
def conveter_aten_pow_int(
        a: int,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.int(int a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.pow.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.complex)
def conveter_aten_pow_complex(
        a: complex,
        b: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.complex(complex a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.pow.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.float)
def conveter_aten_pow_float(
        a: float,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.float(float a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.pow.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.int_float)
def conveter_aten_pow_int_float(
        a: int,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.int_float(int a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.pow.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.float_int)
def conveter_aten_pow_float_int(
        a: float,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.float_int(float a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.pow.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.float_complex)
def conveter_aten_pow_float_complex(
        a: float,
        b: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.float_complex(float a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.pow.float_complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.complex_float)
def conveter_aten_pow_complex_float(
        a: complex,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.complex_float(complex a, float b) -> complex """
    raise NotImplementedError("torch.ops.aten.pow.complex_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.Scalar_Scalar)
def conveter_aten_pow_Scalar_Scalar(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.Scalar_Scalar(Scalar a, Scalar b) -> float """
    raise NotImplementedError("torch.ops.aten.pow.Scalar_Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.pow.int_to_int)
def conveter_aten_pow_int_to_int(
        a: int,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::pow.int_to_int(int a, int b) -> int """
    raise NotImplementedError("torch.ops.aten.pow.int_to_int ge converter is not implement!")


