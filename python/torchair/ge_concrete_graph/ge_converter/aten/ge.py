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


@register_fx_node_ge_converter(torch.ops.aten.ge.Tensor)
def conveter_aten_ge_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::ge.Tensor(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.ge.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.Scalar)
def conveter_aten_ge_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::ge.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.ge.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.Scalar_out)
def conveter_aten_ge_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.ge.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.Tensor_out)
def conveter_aten_ge_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.ge.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.int)
def conveter_aten_ge_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::ge.int(int a, int b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.float)
def conveter_aten_ge_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::ge.float(float a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.int_float)
def conveter_aten_ge_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::ge.int_float(int a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.int_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.float_int)
def conveter_aten_ge_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::ge.float_int(float a, int b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.float_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.default)
def conveter_aten_ge_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::ge(Scalar a, Scalar b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ge.str)
def conveter_aten_ge_str(a: str, b: str, meta_outputs: TensorSpec = None):
    """NB: aten::ge.str(str a, str b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ge.str ge_converter is not implemented!")
