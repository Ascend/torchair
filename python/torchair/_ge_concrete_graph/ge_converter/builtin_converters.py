from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)
import operator
import math
from packaging import version

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec
from torchair.ge._ge_graph import is_sym
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import F32, F16, I32, Support


@register_fx_node_ge_converter(operator.add)
def conveter_operator_add(
    self: Union[Number, Tensor],
    other: Union[Number, Tensor],
    meta_outputs: TensorSpec = None
):
    if all(not isinstance(x, Tensor) for x in (self, other)):
        return self + other
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Add(self, other)


@register_fx_node_ge_converter(operator.sub)
def conveter_operator_sub(
    self: Union[Number, Tensor],
    other: Union[Number, Tensor],
    meta_outputs: TensorSpec = None
):
    if all(not isinstance(x, Tensor) for x in (self, other)):
        return self - other
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Sub(self, other)


@register_fx_node_ge_converter(operator.mul)
def conveter_operator_mul(
    self: Union[Number, Tensor],
    other: Union[Number, Tensor],
    meta_outputs: TensorSpec = None
):
    if all(not isinstance(x, Tensor) for x in (self, other)):
        return self * other
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Mul(self, other)


@register_fx_node_ge_converter(operator.floordiv)
def conveter_operator_floordiv(
    self: Union[Number, Tensor],
    other: Union[Number, Tensor],
    meta_outputs: TensorSpec = None
):
    if all(not isinstance(x, Tensor) for x in (self, other)):
        return self // other
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.FloorDiv(self, other)


@register_fx_node_ge_converter(operator.truediv)
def conveter_operator_truediv(
    self: Union[Number, Tensor],
    other: Union[Number, Tensor],
    meta_outputs: TensorSpec = None
):
    if all(not isinstance(x, Tensor) for x in (self, other)):
        return self / other
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Div(self, other)


@register_fx_node_ge_converter(operator.pow)
def conveter_operator_pow(
        self: Union[Number, Tensor],
        other: Union[Number, Tensor],
        meta_outputs: TensorSpec = None
):
    if all(not isinstance(x, Tensor) for x in (self, other)):
        return self ** other
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Pow(self, other)


@register_fx_node_ge_converter(torch.sym_float)
def conveter_sym_float(
    self: Union[Number, Tensor],
    meta_outputs: TensorSpec = None
):
    if not isinstance(self, Tensor):
        return float(self)
    return ge.Cast(self, dst_type=meta_outputs.dtype)


@declare_supported([
    Support(32),
    Support(F16(1)),
    Support(I32(1)),
])
@register_fx_node_ge_converter(math.floor)
def conveter_math_floor(
    self: Union[Number, Tensor],
    meta_outputs: TensorSpec = None
):
    if not isinstance(self, Tensor):
        return math.floor(self)
    return dtype_promote(ge.Floor(self), target_dtype=meta_outputs.dtype)


@declare_supported([
    Support(F32(2, 2), F32(2, 2)),
    Support(F16(2, 2), F16(2, 2)),
    Support(F16(2, 2), F32(2, 2)),
    Support(F16(4, 3), I32(4, 3)),
])
@register_fx_node_ge_converter(operator.mod)
def conveter_operator_pow(
        self: Union[Number, Tensor],
        other: Union[Number, Tensor],
        meta_outputs: TensorSpec = None
):
    if all(not isinstance(x, Tensor) for x in (self, other)):
        return self % other
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Mod(self, other)


@declare_supported([
    Support(32.34),
    Support(32.76),
    Support(F16(1)),
    Support(I32(1)),
])
@register_fx_node_ge_converter(math.ceil)
def conveter_math_ceil(
        self: Union[Number, Tensor],
        meta_outputs: TensorSpec = None
):
    if not isinstance(self, Tensor):
        return math.ceil(self)
    return dtype_promote(ge.Ceil(self), target_dtype=meta_outputs.dtype)


if version.parse(torch.__version__) >= version.parse("2.6.0"):
    if hasattr(torch, "sym_sum"):
        @declare_supported([
            Support([F32(3, 2), F32(3, 2), F16(3, 2)]),
            Support([F32(3, 2), F32(3, 2)]),
            Support([3, 4, 5])
        ])
        @register_fx_node_ge_converter(torch.sym_sum)
        def converter_aten_sym_sum(inputs, meta_outputs: TensorSpec = None):
            if not isinstance(inputs, Tensor):
                return sum(inputs)

            return ge.ReduceSum(inputs, ge.Const(0, DataType.DT_INT64))


@declare_supported([
    Support(F16(8)),
    Support(I32(3)),
    Support(4),
])
@register_fx_node_ge_converter(operator.neg)
def conveter_operator_neg(
        self: Union[Number, Tensor],
        meta_outputs: TensorSpec = None
):
    return ge.Neg(self)
