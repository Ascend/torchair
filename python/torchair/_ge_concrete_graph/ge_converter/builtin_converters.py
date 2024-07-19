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
import operator
import math

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.ge._ge_graph import is_sym
from torchair._ge_concrete_graph.utils import dtype_promote


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


@register_fx_node_ge_converter(math.floor)
def conveter_math_floor(
    self: Union[Number, Tensor],
    meta_outputs: TensorSpec = None
):
    if not isinstance(self, Tensor):
        return math.floor(self)
    return ge.Floor(self)