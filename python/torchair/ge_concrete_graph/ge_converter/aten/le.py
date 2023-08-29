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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote

@declare_supported([
    Support(F32(2, 3), F32(2, 3)),
])
@register_fx_node_ge_converter(torch.ops.aten.le.Tensor)
def conveter_aten_le_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::le.Tensor(Tensor self, Tensor other) -> Tensor"""
    return ge.LessEqual(self, other)


@register_fx_node_ge_converter(torch.ops.aten.le.Scalar)
def conveter_aten_le_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::le.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.le.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.le.Scalar_out)
def conveter_aten_le_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.le.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.le.Tensor_out)
def conveter_aten_le_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.le.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.le.int)
def conveter_aten_le_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::le.int(int a, int b) -> bool"""
    raise NotImplementedError("torch.ops.aten.le.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.le.float)
def conveter_aten_le_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::le.float(float a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.le.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.le.int_float)
def conveter_aten_le_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::le.int_float(int a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.le.int_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.le.float_int)
def conveter_aten_le_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::le.float_int(float a, int b) -> bool"""
    raise NotImplementedError("torch.ops.aten.le.float_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.le.default)
def conveter_aten_le_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::le(Scalar a, Scalar b) -> bool"""
    raise NotImplementedError("torch.ops.aten.le.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.le.str)
def conveter_aten_le_str(a: str, b: str, meta_outputs: TensorSpec = None):
    """NB: aten::le.str(str a, str b) -> bool"""
    raise NotImplementedError("torch.ops.aten.le.str ge_converter is not implemented!")
