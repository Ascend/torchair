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
from torch import Generator, contiguous_format, inf, memory_format, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.mean.default)
def conveter_aten_mean_default(
    self: Tensor, *, dtype: Optional[int] = None, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.mean.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.mean.dim)
def conveter_aten_mean_dim(
    self: Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    if dtype is not None:
        raise NotImplementedError(
            "torch.ops.aten.mean.dim have some unprocessed parameters or cases, "
            "dtype = {}".format(dtype))

    return ge.ReduceMean(self, dim, keep_dims=keepdim)


@register_fx_node_ge_converter(torch.ops.aten.mean.names_dim)
def conveter_aten_mean_names_dim(
    self: Tensor,
    dim: List[str],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::mean.names_dim(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.mean.names_dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.mean.names_out)
def conveter_aten_mean_names_out(
    self: Tensor,
    dim: List[str],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::mean.names_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.mean.names_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.mean.out)
def conveter_aten_mean_out(
    self: Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.mean.out ge_converter is not implemented!")
