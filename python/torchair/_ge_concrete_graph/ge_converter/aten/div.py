from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support, T
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(2, 2), F32(2, 2)),
        Support(F32(2, 2), F32(1, 2)),
        Support(F32(2, 2), F16(2, 1)),
        Support(U8(2, 2), 10),
        Support(T([0, -6], dtype=torch.int32), -2),
        Support(T([[5, -3]], dtype=torch.int16), 0),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.div.Tensor)
def conveter_aten_div_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::div.Tensor(Tensor self, Tensor other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.RealDiv(self, other)


@declare_supported(
    [
        Support(F32(2, 2), int(3)),
        Support(F32(2, 2), float(3.2)),
        Support(F32(2, 2), float(3.9)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.div.Scalar)
def conveter_aten_div_Scalar(
        self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::div.Scalar(Tensor self, Scalar other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.RealDiv(self, other)


@declare_supported([
    Support(F32(20), 42, rounding_mode="floor"),
    Support(F32(20), 42, rounding_mode="trunc"),
    Support(F32(20), 42, rounding_mode=None),
    Support(I16(20), 42, rounding_mode="trunc"),
    Support(I16(20), I16(20, value_range=(1, 100)), rounding_mode="trunc"),
])
@register_fx_node_ge_converter(torch.ops.aten.div.Tensor_mode)
def conveter_aten_div_Tensor_mode(
        self: Tensor,
        other: Tensor,
        *,
        rounding_mode: Optional[str],
        meta_outputs: TensorSpec = None
):
    """NB: aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    if rounding_mode == "floor":
        output = ge.FloorDiv(self, other)
    elif rounding_mode == "trunc":
        output = ge.RealDiv(self, other)
        dtype = meta_outputs.dtype
        if dtype in [DataType.DT_INT64, DataType.DT_BOOL, DataType.DT_INT16]:
            output = ge.Cast(output, dst_type=DataType.DT_FLOAT)
            output = ge.Trunc(output)
            output = ge.Cast(output, dst_type=dtype)
        else:
            output = ge.Trunc(output)
    else:
        output = ge.RealDiv(self, other)
    return output


@register_fx_node_ge_converter(torch.ops.aten.div.Scalar_mode)
def conveter_aten_div_Scalar_mode(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        rounding_mode: Optional[str],
        meta_outputs: TensorSpec = None
):
    """NB: aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor"""
    raise AssertionError(
        "torch.ops.aten.div.Scalar_mode is redundant before pytorch 2.1.0, might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.div.out)
def conveter_aten_div_out(
        self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise AssertionError(
        "torch.ops.aten.div.out is redundant before pytorch 2.1.0, might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.div.out_mode)
def conveter_aten_div_out_mode(
        self: Tensor,
        other: Tensor,
        *,
        rounding_mode: Optional[str],
        out: Tensor = None,
        meta_outputs: TensorSpec = None
):
    """NB: aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)"""
    raise AssertionError(
        "torch.ops.aten.div.out_mode is redundant before pytorch 2.1.0, might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.div.Scalar_out)
def conveter_aten_div_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: TensorSpec = None
):
    """NB: aten::div.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise AssertionError(
        "torch.ops.aten.div.Scalar_out is redundant before pytorch 2.1.0, might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.div.Scalar_mode_out)
def conveter_aten_div_Scalar_mode_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        rounding_mode: Optional[str],
        out: Tensor = None,
        meta_outputs: TensorSpec = None
):
    """NB: aten::div.Scalar_mode_out(Tensor self, Scalar other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)"""
    raise AssertionError(
        "torch.ops.aten.div.Scalar_mode_out is redundant before pytorch 2.1.0, might be supported in future version.")


@register_fx_node_ge_converter(torch.ops.aten.div.int)
def conveter_aten_div_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::div.int(int a, int b) -> float"""
    raise AssertionError("torch.ops.aten.div.int is redundant. please use torch.ops.aten.div.Tensor instead")


@register_fx_node_ge_converter(torch.ops.aten.div.complex)
def conveter_aten_div_complex(a: complex, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::div.complex(complex a, complex b) -> complex"""
    raise AssertionError("torch.ops.aten.div.complex is redundant. please use torch.ops.aten.div.Tensor instead")


@register_fx_node_ge_converter(torch.ops.aten.div.float)
def conveter_aten_div_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::div.float(float a, float b) -> float"""
    raise AssertionError("torch.ops.aten.div.float is redundant. please use torch.ops.aten.div.Tensor instead")


@register_fx_node_ge_converter(torch.ops.aten.div.default)
def conveter_aten_div_default(
        a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::div(Scalar a, Scalar b) -> float"""
    raise AssertionError("torch.ops.aten.div.default is redundant. please use torch.ops.aten.div.Tensor instead")
