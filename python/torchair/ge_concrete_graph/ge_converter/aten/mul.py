import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
from torchair.ge_concrete_graph.utils import dtype_promote
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


@register_fx_node_ge_converter(torch.ops.aten.mul.Tensor)
def conveter_aten_mul_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::mul.Tensor(Tensor self, Tensor other) -> Tensor """
    self, other = dtype_promote(self, other, target_dtype = meta_outputs.dtype)
    return ge.Mul(self, other)


@register_fx_node_ge_converter(torch.ops.aten.mul.Scalar)
def conveter_aten_mul_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::mul.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.mul.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.out)
def conveter_aten_mul_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.mul.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.Scalar_out)
def conveter_aten_mul_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::mul.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.mul.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.int)
def conveter_aten_mul_int(
        a: int,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::mul.int(int a, int b) -> int """
    raise NotImplementedError("torch.ops.aten.mul.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.complex)
def conveter_aten_mul_complex(
        a: complex,
        b: complex,
        meta_outputs: Any = None):
    """ NB: aten::mul.complex(complex a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.mul.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.float)
def conveter_aten_mul_float(
        a: float,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::mul.float(float a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.mul.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.int_complex)
def conveter_aten_mul_int_complex(
        a: int,
        b: complex,
        meta_outputs: Any = None):
    """ NB: aten::mul.int_complex(int a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.mul.int_complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.complex_int)
def conveter_aten_mul_complex_int(
        a: complex,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::mul.complex_int(complex a, int b) -> complex """
    raise NotImplementedError("torch.ops.aten.mul.complex_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.float_complex)
def conveter_aten_mul_float_complex(
        a: float,
        b: complex,
        meta_outputs: Any = None):
    """ NB: aten::mul.float_complex(float a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.mul.float_complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.complex_float)
def conveter_aten_mul_complex_float(
        a: complex,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::mul.complex_float(complex a, float b) -> complex """
    raise NotImplementedError("torch.ops.aten.mul.complex_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.int_float)
def conveter_aten_mul_int_float(
        a: int,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::mul.int_float(int a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.mul.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.float_int)
def conveter_aten_mul_float_int(
        a: float,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::mul.float_int(float a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.mul.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mul.default)
def conveter_aten_mul_default(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::mul(Scalar a, Scalar b) -> Scalar """
    raise NotImplementedError("torch.ops.aten.mul.default ge converter is not implement!")


