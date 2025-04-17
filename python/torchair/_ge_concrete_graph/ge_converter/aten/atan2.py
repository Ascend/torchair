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


@register_fx_node_ge_converter(torch.ops.aten.atan2.default)
def conveter_aten_atan2_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::atan2(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.atan2.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.out)
def conveter_aten_atan2_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.atan2.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.int)
def conveter_aten_atan2_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::atan2.int(int a, int b) -> float"""
    raise NotImplementedError("torch.ops.aten.atan2.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.float)
def conveter_aten_atan2_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::atan2.float(float a, float b) -> float"""
    raise NotImplementedError("torch.ops.aten.atan2.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.int_float)
def conveter_aten_atan2_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::atan2.int_float(int a, float b) -> float"""
    raise NotImplementedError("torch.ops.aten.atan2.int_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.float_int)
def conveter_aten_atan2_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::atan2.float_int(float a, int b) -> float"""
    raise NotImplementedError("torch.ops.aten.atan2.float_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan2.Scalar_Scalar)
def conveter_aten_atan2_Scalar_Scalar(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::atan2.Scalar_Scalar(Scalar a, Scalar b) -> float"""
    raise NotImplementedError("torch.ops.aten.atan2.Scalar_Scalar ge_converter is not implemented!")
