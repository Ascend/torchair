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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Tensor)
def conveter_aten_bitwise_and_Tensor(
    self: Tensor, other: Tensor, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bitwise_and.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Scalar)
def conveter_aten_bitwise_and_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bitwise_and.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Scalar_Tensor)
def conveter_aten_bitwise_and_Scalar_Tensor(
    self: Union[Number, Tensor], other: Tensor, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::bitwise_and.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.bitwise_and.Scalar_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Tensor_out)
def conveter_aten_bitwise_and_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_and.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Scalar_out)
def conveter_aten_bitwise_and_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_and.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_and.Scalar_Tensor_out)
def conveter_aten_bitwise_and_Scalar_Tensor_out(
    self: Union[Number, Tensor],
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::bitwise_and.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_and.Scalar_Tensor_out ge_converter is not implemented!")
