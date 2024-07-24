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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, torch_type_to_ge_type, declare_supported, \
    torch_type_to_ge_type
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote

@declare_supported([
    Support(BOOL(2, 2), BOOL(2, 2)),
    Support(I32(2, 2), I32(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Tensor)
def conveter_aten_bitwise_and_Tensor(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    if self.dtype == torch_type_to_ge_type(torch.bool):
        output = ge.LogicalAnd(self, other)
    else:
        output = ge.BitwiseAnd(self, other)
    return output


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Scalar)
def conveter_aten_bitwise_and_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise RuntimeError("torch.ops.aten.bitwise_and.Scalar ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Scalar_Tensor)
def conveter_aten_bitwise_and_Scalar_Tensor(
    self: Union[Number, Tensor], other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_and.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    raise RuntimeError("torch.ops.aten.bitwise_and.Scalar_Tensor ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Tensor_out)
def conveter_aten_bitwise_and_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.bitwise_and.Tensor_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Scalar_out)
def conveter_aten_bitwise_and_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.bitwise_and.Scalar_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Scalar_Tensor_out)
def conveter_aten_bitwise_and_Scalar_Tensor_out(
    self: Union[Number, Tensor],
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_and.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.bitwise_and.Scalar_Tensor_out ge_converter is not supported!")
