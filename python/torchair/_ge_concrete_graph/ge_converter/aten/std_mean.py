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


@register_fx_node_ge_converter(torch.ops.aten.std_mean.default)
def conveter_aten_std_mean_default(
    self: Tensor, unbiased: bool = True, meta_outputs: TensorSpec = None
):
    """NB: aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.std_mean.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std_mean.dim)
def conveter_aten_std_mean_dim(
    self: Tensor,
    dim: Optional[List[int]],
    unbiased: bool = True,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::std_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.std_mean.dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std_mean.correction)
def conveter_aten_std_mean_correction(
    self: Tensor,
    dim: Optional[List[int]] = None,
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::std_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.std_mean.correction ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std_mean.names_dim)
def conveter_aten_std_mean_names_dim(
    self: Tensor,
    dim: List[str],
    unbiased: bool = True,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::std_mean.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.std_mean.names_dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std_mean.correction_names)
def conveter_aten_std_mean_correction_names(
    self: Tensor,
    dim: List[str],
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::std_mean.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.std_mean.correction_names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std_mean.correction_out)
def conveter_aten_std_mean_correction_out(
    self: Tensor,
    dim: Optional[List[int]] = None,
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    out0: Tensor = None,
    out1: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::std_mean.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))"""
    raise NotImplementedError("torch.ops.aten.std_mean.correction_out ge_converter is not implemented!")
