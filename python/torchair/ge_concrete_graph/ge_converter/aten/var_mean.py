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


@register_fx_node_ge_converter(torch.ops.aten.var_mean.default)
def conveter_aten_var_mean_default(
    self: Tensor, unbiased: bool = True, meta_outputs: TensorSpec = None
):
    """NB: aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.var_mean.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var_mean.dim)
def conveter_aten_var_mean_dim(
    self: Tensor,
    dim: Optional[List[int]],
    unbiased: bool = True,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::var_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.var_mean.dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var_mean.correction)
def conveter_aten_var_mean_correction(
    self: Tensor,
    dim: Optional[List[int]] = None,
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::var_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.var_mean.correction ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var_mean.names_dim)
def conveter_aten_var_mean_names_dim(
    self: Tensor,
    dim: List[str],
    unbiased: bool = True,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::var_mean.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.var_mean.names_dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var_mean.correction_names)
def conveter_aten_var_mean_correction_names(
    self: Tensor,
    dim: List[str],
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::var_mean.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.var_mean.correction_names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.var_mean.correction_out)
def conveter_aten_var_mean_correction_out(
    self: Tensor,
    dim: Optional[List[int]] = None,
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    out0: Tensor = None,
    out1: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::var_mean.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))"""
    raise NotImplementedError("torch.ops.aten.var_mean.correction_out ge_converter is not implemented!")
