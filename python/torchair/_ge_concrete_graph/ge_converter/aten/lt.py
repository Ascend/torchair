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
    Support(F32(2, 2), F32(2, 2)),
    Support(I8(2, 2), F32(2, 2))
])
@register_fx_node_ge_converter(torch.ops.aten.lt.Tensor)
def conveter_aten_lt_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::lt.Tensor(Tensor self, Tensor other) -> Tensor"""
    """This geir not implement bool dtype input, and dtype must be same"""
    return ge.Less(self, other)


@declare_supported([
    Support(F32(2, 2), 1),
    Support(I8(2, 2), 1)
])
@register_fx_node_ge_converter(torch.ops.aten.lt.Scalar)
def conveter_aten_lt_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::lt.Scalar(Tensor self, Scalar other) -> Tensor"""
    """This geir not implement bool dtype input"""
    return ge.Less(self, ge.Cast(other, dst_type=self.dtype))


@register_fx_node_ge_converter(torch.ops.aten.lt.Scalar_out)
def conveter_aten_lt_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.lt.Scalar_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.lt.Tensor_out)
def conveter_aten_lt_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.lt.Tensor_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.lt.int)
def conveter_aten_lt_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::lt.int(int a, int b) -> bool"""
    raise RuntimeError("torch.ops.aten.lt.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.lt.float)
def conveter_aten_lt_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::lt.float(float a, float b) -> bool"""
    raise RuntimeError("torch.ops.aten.lt.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.lt.int_float)
def conveter_aten_lt_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::lt.int_float(int a, float b) -> bool"""
    raise RuntimeError("torch.ops.aten.lt.int_float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.lt.float_int)
def conveter_aten_lt_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::lt.float_int(float a, int b) -> bool"""
    raise RuntimeError("torch.ops.aten.lt.float_int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.lt.default)
def conveter_aten_lt_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::lt(Scalar a, Scalar b) -> bool"""
    raise RuntimeError("torch.ops.aten.lt.default ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.lt.str)
def conveter_aten_lt_str(a: str, b: str, meta_outputs: TensorSpec = None):
    """NB: aten::lt.str(str a, str b) -> bool"""
    raise RuntimeError("torch.ops.aten.lt.str ge_converter is not supported!")
