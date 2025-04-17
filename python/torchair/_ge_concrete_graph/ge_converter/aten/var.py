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


@register_fx_node_ge_converter(torch.ops.aten.var.default)
def conveter_aten_var_default(
    self: Tensor, unbiased: bool = True, meta_outputs: TensorSpec = None
):
    """NB: aten::var(Tensor self, bool unbiased=True) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.var.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var.dim)
def conveter_aten_var_dim(
    self: Tensor,
    dim: Optional[List[int]],
    unbiased: bool = True,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.var.dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var.correction)
def conveter_aten_var_correction(
    self: Tensor,
    dim: Optional[List[int]] = None,
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.var.correction ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var.names_dim)
def conveter_aten_var_names_dim(
    self: Tensor,
    dim: List[str],
    unbiased: bool = True,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::var.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.var.names_dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var.names_out)
def conveter_aten_var_names_out(
    self: Tensor,
    dim: List[str],
    unbiased: bool = True,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::var.names_out(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.var.names_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var.out)
def conveter_aten_var_out(
    self: Tensor,
    dim: Optional[List[int]],
    unbiased: bool = True,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::var.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.var.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var.correction_out)
def conveter_aten_var_correction_out(
    self: Tensor,
    dim: Optional[List[int]] = None,
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::var.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.var.correction_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var.correction_names)
def conveter_aten_var_correction_names(
    self: Tensor,
    dim: List[str],
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::var.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.var.correction_names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var.correction_names_out)
def conveter_aten_var_correction_names_out(
    self: Tensor,
    dim: List[str],
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::var.correction_names_out(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.var.correction_names_out ge_converter is not implemented!")
