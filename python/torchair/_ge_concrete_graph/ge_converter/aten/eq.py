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
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.eq.Tensor)
def conveter_aten_eq_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::eq.Tensor(Tensor self, Tensor other) -> Tensor"""
    if self.dtype != other.dtype:
        raise AssertionError(f"Inputs data type mismatch {other.dtype} vs. {other.dtype}")
    return ge.Equal(self, other)


@declare_supported(
    [
        Support(F32(2, 2), 0),
        Support(F32(2, 2), 1.0),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.eq.Scalar)
def conveter_aten_eq_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::eq.Scalar(Tensor self, Scalar other) -> Tensor"""
    other = dtype_promote(other, target_dtype=self.dtype)
    return ge.Equal(self, other)


@register_fx_node_ge_converter(torch.ops.aten.eq.Scalar_out)
def conveter_aten_eq_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.eq.Scalar_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.Tensor_out)
def conveter_aten_eq_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.eq.Tensor_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.int_list)
def conveter_aten_eq_int_list(a: List[int], b: List[int], meta_outputs: TensorSpec = None):
    """NB: aten::eq.int_list(int[] a, int[] b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.int_list ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.device)
def conveter_aten_eq_device(a: Device, b: Device, meta_outputs: TensorSpec = None):
    """NB: aten::eq.device(Device a, Device b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.device ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.bool)
def conveter_aten_eq_bool(a: bool, b: bool, meta_outputs: TensorSpec = None):
    """NB: aten::eq.bool(bool a, bool b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.bool ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.int)
def conveter_aten_eq_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::eq.int(int a, int b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.complex)
def conveter_aten_eq_complex(a: complex, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::eq.complex(complex a, complex b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.complex ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.float)
def conveter_aten_eq_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::eq.float(float a, float b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.int_float)
def conveter_aten_eq_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::eq.int_float(int a, float b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.int_float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.float_int)
def conveter_aten_eq_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::eq.float_int(float a, int b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.float_int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.float_complex)
def conveter_aten_eq_float_complex(a: float, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::eq.float_complex(float a, complex b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.float_complex ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.complex_float)
def conveter_aten_eq_complex_float(a: complex, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::eq.complex_float(complex a, float b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.complex_float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.default)
def conveter_aten_eq_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::eq(Scalar a, Scalar b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.default ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.str)
def conveter_aten_eq_str(a: str, b: str, meta_outputs: TensorSpec = None):
    """NB: aten::eq.str(str a, str b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.str ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.float_list)
def conveter_aten_eq_float_list(
    a: List[float], b: List[float], meta_outputs: TensorSpec = None
):
    """NB: aten::eq.float_list(float[] a, float[] b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.float_list ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.Tensor_list)
def conveter_aten_eq_Tensor_list(
    a: List[Tensor], b: List[Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::eq.Tensor_list(Tensor[] a, Tensor[] b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.Tensor_list ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.bool_list)
def conveter_aten_eq_bool_list(a: List[bool], b: List[bool], meta_outputs: TensorSpec = None):
    """NB: aten::eq.bool_list(bool[] a, bool[] b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.bool_list ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.eq.str_list)
def conveter_aten_eq_str_list(a: List[str], b: List[str], meta_outputs: TensorSpec = None):
    """NB: aten::eq.str_list(str[] a, str[] b) -> bool"""
    raise RuntimeError("torch.ops.aten.eq.str_list ge_converter is not supported!")
