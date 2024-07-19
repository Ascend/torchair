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


@declare_supported(
    [
        Support(F32(2, 2), F32(2, 2)),
        Support(F32(2, 2), F32(2, 1)),
        Support(F32(2, 2), F16(2, 1)),
        Support(F32(2, 2), F16(2, 2), alpha=2),
        Support(F32(2, 2), 2.0),
        Support(F32(2, 2), 2),
        Support(F32(2, 2), 2, alpha=2.0),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.add.Tensor)
def conveter_aten_add_Tensor(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    if not isinstance(alpha, Tensor) and alpha == 1:
        # just for better permance
        self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
        return ge.Add(self, other)
    else:
        self, other, alpha = dtype_promote(self, other, alpha, target_dtype=meta_outputs.dtype)
        return ge.AxpyV2(self, other, alpha)


@register_fx_node_ge_converter(torch.ops.aten.add.Scalar)
def conveter_aten_add_Scalar(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"""
    if not isinstance(alpha, Tensor) and alpha == 1:
        # just for better permance
        self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
        return ge.Add(self, other)
    else:
        self, other, alpha = dtype_promote(self, other, alpha, target_dtype=meta_outputs.dtype)
        return ge.AxpyV2(self, other, alpha)


@register_fx_node_ge_converter(torch.ops.aten.add.out)
def conveter_aten_add_out(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.add.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.Scalar_out)
def conveter_aten_add_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::add.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.add.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.str)
def conveter_aten_add_str(a: str, b: str, meta_outputs: TensorSpec = None):
    """NB: aten::add.str(str a, str b) -> str"""
    raise NotImplementedError("torch.ops.aten.add.str ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.int)
def conveter_aten_add_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::add.int(int a, int b) -> int"""
    raise NotImplementedError("torch.ops.aten.add.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.complex)
def conveter_aten_add_complex(a: complex, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::add.complex(complex a, complex b) -> complex"""
    raise NotImplementedError("torch.ops.aten.add.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.float)
def conveter_aten_add_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::add.float(float a, float b) -> float"""
    raise NotImplementedError("torch.ops.aten.add.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.int_complex)
def conveter_aten_add_int_complex(a: int, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::add.int_complex(int a, complex b) -> complex"""
    raise NotImplementedError("torch.ops.aten.add.int_complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.complex_int)
def conveter_aten_add_complex_int(a: complex, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::add.complex_int(complex a, int b) -> complex"""
    raise NotImplementedError("torch.ops.aten.add.complex_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.float_complex)
def conveter_aten_add_float_complex(a: float, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::add.float_complex(float a, complex b) -> complex"""
    raise NotImplementedError("torch.ops.aten.add.float_complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.complex_float)
def conveter_aten_add_complex_float(a: complex, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::add.complex_float(complex a, float b) -> complex"""
    raise NotImplementedError("torch.ops.aten.add.complex_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.int_float)
def conveter_aten_add_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::add.int_float(int a, float b) -> float"""
    raise NotImplementedError("torch.ops.aten.add.int_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.float_int)
def conveter_aten_add_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::add.float_int(float a, int b) -> float"""
    raise NotImplementedError("torch.ops.aten.add.float_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.add.default)
def conveter_aten_add_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::add(Scalar a, Scalar b) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.add.default ge_converter is not implemented!")
