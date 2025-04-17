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
    Support, BF16
from torchair._ge_concrete_graph.utils import dtype_promote, DataType


def _get_mul_compute_dtype(self, other, output_dtype):
    if isinstance(self, Tensor) and isinstance(other, Tensor) and \
            self.dtype == DataType.DT_BOOL and other.dtype == DataType.DT_BOOL:
        return DataType.DT_UINT8
    return output_dtype


@declare_supported([
    Support(F32(2, 2), F32(2, 2)),
    Support(F16(2, 2), F16(2, 2)),
    Support(F32(2, 2), BOOL(2, 2)),
    Support(BOOL(2, 2), F32(2, 2)),
    Support(BOOL(2, 2), BOOL(2, 2)),
    Support(BOOL(2, 2), 3),
    Support(3, BOOL(2, 2)),
    Support(F16(2, 2), 2),
    Support(BF16(2, 2), 2),
])
@register_fx_node_ge_converter(torch.ops.aten.mul.Tensor)
def conveter_aten_mul_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::mul.Tensor(Tensor self, Tensor other) -> Tensor"""
    """Mul operater doesn't support the input format of (bool, bool) till 2023/11/17"""
    if not isinstance(other, Tensor) and isinstance(self, Tensor) and \
            self.dtype in [DataType.DT_FLOAT16, DataType.DT_BF16]:
        output = ge.Muls(self, value=other)
        return dtype_promote(output, target_dtype=meta_outputs.dtype)

    compute_dtype = _get_mul_compute_dtype(self, other, meta_outputs.dtype)
    self, other = dtype_promote(self, other, target_dtype=compute_dtype)
    output = ge.Mul(self, other)
    output = dtype_promote(output, target_dtype=meta_outputs.dtype)
    return output


@declare_supported([
    Support(F32(2, 2), 2),
])
@register_fx_node_ge_converter(torch.ops.aten.mul.Scalar)
def conveter_aten_mul_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::mul.Scalar(Tensor self, Scalar other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Mul(self, other)


@register_fx_node_ge_converter(torch.ops.aten.mul.out)
def conveter_aten_mul_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.mul.out ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.Scalar_out)
def conveter_aten_mul_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::mul.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.mul.Scalar_out ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.int)
def conveter_aten_mul_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::mul.int(int a, int b) -> int"""
    raise RuntimeError("torch.ops.aten.mul.int ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.complex)
def conveter_aten_mul_complex(a: complex, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::mul.complex(complex a, complex b) -> complex"""
    raise RuntimeError("torch.ops.aten.mul.complex ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.float)
def conveter_aten_mul_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::mul.float(float a, float b) -> float"""
    raise RuntimeError("torch.ops.aten.mul.float ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.int_complex)
def conveter_aten_mul_int_complex(a: int, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::mul.int_complex(int a, complex b) -> complex"""
    raise RuntimeError("torch.ops.aten.mul.int_complex ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.complex_int)
def conveter_aten_mul_complex_int(a: complex, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::mul.complex_int(complex a, int b) -> complex"""
    raise RuntimeError("torch.ops.aten.mul.complex_int ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.float_complex)
def conveter_aten_mul_float_complex(a: float, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::mul.float_complex(float a, complex b) -> complex"""
    raise RuntimeError("torch.ops.aten.mul.float_complex ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.complex_float)
def conveter_aten_mul_complex_float(a: complex, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::mul.complex_float(complex a, float b) -> complex"""
    raise RuntimeError("torch.ops.aten.mul.complex_float ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.int_float)
def conveter_aten_mul_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::mul.int_float(int a, float b) -> float"""
    raise RuntimeError("torch.ops.aten.mul.int_float ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.float_int)
def conveter_aten_mul_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::mul.float_int(float a, int b) -> float"""
    raise RuntimeError("torch.ops.aten.mul.float_int ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.mul.default)
def conveter_aten_mul_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::mul(Scalar a, Scalar b) -> Scalar"""
    raise RuntimeError("torch.ops.aten.mul.default ge_converter is redundant before pytorch 2.1.0!")
