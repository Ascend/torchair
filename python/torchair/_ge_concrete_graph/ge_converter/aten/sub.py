from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
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
    Support(F32(2, 2), F32(2, 2), alpha=2.0),
    Support(F32(2, 2), F16(2, 2), alpha=2.0),
    Support(F16(2, 2), I8(2), alpha=2.0),
    Support(I8(2, 2), I16(2, 2), alpha=2),
    Support(I16(2, 2), I8(2), alpha=2),
])
@register_fx_node_ge_converter(torch.ops.aten.sub.Tensor)
def conveter_aten_sub_Tensor(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    if alpha != 1:
        alpha = ge.Const(alpha, dtype=other.dtype)
        other = ge.Mul(other, alpha)
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Sub(self, other)


@register_fx_node_ge_converter(torch.ops.aten.sub.Scalar)
def conveter_aten_sub_Scalar(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"""
    raise RuntimeError("torch.ops.aten.sub.Scalar ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.out)
def conveter_aten_sub_out(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.sub.out ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.Scalar_out)
def conveter_aten_sub_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.sub.Scalar_out ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.int)
def conveter_aten_sub_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::sub.int(int a, int b) -> int"""
    raise RuntimeError("torch.ops.aten.sub.int ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.complex)
def conveter_aten_sub_complex(a: complex, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::sub.complex(complex a, complex b) -> complex"""
    raise RuntimeError("torch.ops.aten.sub.complex ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.float)
def conveter_aten_sub_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::sub.float(float a, float b) -> float"""
    raise RuntimeError("torch.ops.aten.sub.float ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.int_complex)
def conveter_aten_sub_int_complex(a: int, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::sub.int_complex(int a, complex b) -> complex"""
    raise RuntimeError("torch.ops.aten.sub.int_complex ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.complex_int)
def conveter_aten_sub_complex_int(a: complex, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::sub.complex_int(complex a, int b) -> complex"""
    raise RuntimeError("torch.ops.aten.sub.complex_int ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.float_complex)
def conveter_aten_sub_float_complex(a: float, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::sub.float_complex(float a, complex b) -> complex"""
    raise RuntimeError("torch.ops.aten.sub.float_complex ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.complex_float)
def conveter_aten_sub_complex_float(a: complex, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::sub.complex_float(complex a, float b) -> complex"""
    raise RuntimeError("torch.ops.aten.sub.complex_float ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.int_float)
def conveter_aten_sub_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::sub.int_float(int a, float b) -> float"""
    raise RuntimeError("torch.ops.aten.sub.int_float ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.float_int)
def conveter_aten_sub_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::sub.float_int(float a, int b) -> float"""
    raise RuntimeError("torch.ops.aten.sub.float_int ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.sub.default)
def conveter_aten_sub_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::sub(Scalar a, Scalar b) -> Scalar"""
    raise RuntimeError("torch.ops.aten.sub.default ge_converter is redundant before pytorch 2.1.0!")
