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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.min.other)
def conveter_aten_min_other(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::min.other(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.min.other ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.min.default)
def conveter_aten_min_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::min(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.min.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.min.dim)
def conveter_aten_min_dim(
    self: Tensor, dim: int, keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"""
    raise NotImplementedError("torch.ops.aten.min.dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.min.dim_min)
def conveter_aten_min_dim_min(
    self: Tensor,
    dim: int,
    keepdim: bool = False,
    *,
    min: Tensor = None,
    min_indices: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise NotImplementedError("torch.ops.aten.min.dim_min ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.min.names_dim)
def conveter_aten_min_names_dim(
    self: Tensor, dim: str, keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::min.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)"""
    raise NotImplementedError("torch.ops.aten.min.names_dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.min.names_dim_min)
def conveter_aten_min_names_dim_min(
    self: Tensor,
    dim: str,
    keepdim: bool = False,
    *,
    min: Tensor = None,
    min_indices: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::min.names_dim_min(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise NotImplementedError("torch.ops.aten.min.names_dim_min ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.min.unary_out)
def conveter_aten_min_unary_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::min.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.min.unary_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.min.out)
def conveter_aten_min_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.min.out ge_converter is not implemented!")
