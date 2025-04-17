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


@register_fx_node_ge_converter(torch.ops.aten.polar.default)
def conveter_aten_polar_default(abs: Tensor, angle: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::polar(Tensor abs, Tensor angle) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.polar.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.polar.out)
def conveter_aten_polar_out(
    abs: Tensor, angle: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::polar.out(Tensor abs, Tensor angle, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.polar.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.polar.int)
def conveter_aten_polar_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::polar.int(int a, int b) -> complex"""
    raise NotImplementedError("torch.ops.aten.polar.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.polar.float)
def conveter_aten_polar_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::polar.float(float a, float b) -> complex"""
    raise NotImplementedError("torch.ops.aten.polar.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.polar.int_float)
def conveter_aten_polar_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::polar.int_float(int a, float b) -> complex"""
    raise NotImplementedError("torch.ops.aten.polar.int_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.polar.float_int)
def conveter_aten_polar_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::polar.float_int(float a, int b) -> complex"""
    raise NotImplementedError("torch.ops.aten.polar.float_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.polar.Scalar_Scalar)
def conveter_aten_polar_Scalar_Scalar(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::polar.Scalar_Scalar(Scalar a, Scalar b) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.polar.Scalar_Scalar ge_converter is not implemented!")
