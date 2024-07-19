from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), I32(2)),
    Support(F16(2, 2), I32(2)),
    Support(F64(2, 2), I8(2)),
    Support(I64(2, 2), U8(2)),
])
@register_fx_node_ge_converter(torch.ops.aten.pow.Tensor_Tensor)
def conveter_aten_pow_Tensor_Tensor(
    self: Tensor, exponent: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor"""
    self, exponent = dtype_promote(self, exponent, target_dtype=meta_outputs.dtype)
    return ge.Pow(self, exponent)


@declare_supported([
    Support(F32(2, 2), exponent=3.0),
    Support(F32(2, 2), exponent=2),
])

@register_fx_node_ge_converter(torch.ops.aten.pow.Tensor_Scalar)
def conveter_aten_pow_Tensor_Scalar(
    self: Tensor, exponent: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor"""
    self, exponent = dtype_promote(self, exponent, target_dtype=meta_outputs.dtype)
    return ge.Pow(self, exponent)


@declare_supported([
    Support(10000, F32(32)),
    Support(6, F16(4)),
])
@register_fx_node_ge_converter(torch.ops.aten.pow.Scalar)
def conveter_aten_pow_Scalar(
    self: Union[Number, Tensor], exponent: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor"""
    self, exponent = dtype_promote(self, exponent, target_dtype=meta_outputs.dtype)
    return ge.Pow(self, exponent)


@register_fx_node_ge_converter(torch.ops.aten.pow.Scalar_out)
def conveter_aten_pow_Scalar_out(
    self: Union[Number, Tensor],
    exponent: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.pow.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.Tensor_Scalar_out)
def conveter_aten_pow_Tensor_Scalar_out(
    self: Tensor,
    exponent: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.pow.Tensor_Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.Tensor_Tensor_out)
def conveter_aten_pow_Tensor_Tensor_out(
    self: Tensor, exponent: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.pow.Tensor_Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.int)
def conveter_aten_pow_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::pow.int(int a, int b) -> float"""
    raise NotImplementedError("torch.ops.aten.pow.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.complex)
def conveter_aten_pow_complex(a: complex, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::pow.complex(complex a, complex b) -> complex"""
    raise NotImplementedError("torch.ops.aten.pow.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.float)
def conveter_aten_pow_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::pow.float(float a, float b) -> float"""
    raise NotImplementedError("torch.ops.aten.pow.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.int_float)
def conveter_aten_pow_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::pow.int_float(int a, float b) -> float"""
    raise NotImplementedError("torch.ops.aten.pow.int_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.float_int)
def conveter_aten_pow_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::pow.float_int(float a, int b) -> float"""
    raise NotImplementedError("torch.ops.aten.pow.float_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.float_complex)
def conveter_aten_pow_float_complex(a: float, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::pow.float_complex(float a, complex b) -> complex"""
    raise NotImplementedError("torch.ops.aten.pow.float_complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.complex_float)
def conveter_aten_pow_complex_float(a: complex, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::pow.complex_float(complex a, float b) -> complex"""
    raise NotImplementedError("torch.ops.aten.pow.complex_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.Scalar_Scalar)
def conveter_aten_pow_Scalar_Scalar(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::pow.Scalar_Scalar(Scalar a, Scalar b) -> float"""
    raise NotImplementedError("torch.ops.aten.pow.Scalar_Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pow.int_to_int)
def conveter_aten_pow_int_to_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::pow.int_to_int(int a, int b) -> int"""
    raise NotImplementedError("torch.ops.aten.pow.int_to_int ge_converter is not implemented!")
