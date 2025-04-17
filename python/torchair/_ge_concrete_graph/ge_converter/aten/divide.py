from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.divide.Tensor)
def conveter_aten_divide_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::divide.Tensor(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.divide.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.divide.Scalar)
def conveter_aten_divide_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::divide.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.divide.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.divide.Tensor_mode)
def conveter_aten_divide_Tensor_mode(
    self: Tensor,
    other: Tensor,
    *,
    rounding_mode: Optional[str],
    meta_outputs: TensorSpec = None
):
    """NB: aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.divide.Tensor_mode ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.divide.Scalar_mode)
def conveter_aten_divide_Scalar_mode(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    rounding_mode: Optional[str],
    meta_outputs: TensorSpec = None
):
    """NB: aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.divide.Scalar_mode ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.divide.out)
def conveter_aten_divide_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.divide.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.divide.out_mode)
def conveter_aten_divide_out_mode(
    self: Tensor,
    other: Tensor,
    *,
    rounding_mode: Optional[str],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.divide.out_mode ge_converter is not implemented!")
