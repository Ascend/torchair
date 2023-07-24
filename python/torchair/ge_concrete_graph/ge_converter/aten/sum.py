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


@register_fx_node_ge_converter(torch.ops.aten.sum.dim_IntList)
def conveter_aten_sum_dim_IntList(
    self: Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.sum.dim_IntList ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sum.default)
def conveter_aten_sum_default(
    self: Tensor, *, dtype: Optional[int] = None, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.sum.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sum.dim_DimnameList)
def conveter_aten_sum_dim_DimnameList(
    self: Tensor,
    dim: List[str],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::sum.dim_DimnameList(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.sum.dim_DimnameList ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sum.DimnameList_out)
def conveter_aten_sum_DimnameList_out(
    self: Tensor,
    dim: List[str],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::sum.DimnameList_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sum.DimnameList_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sum.IntList_out)
def conveter_aten_sum_IntList_out(
    self: Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sum.IntList_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sum.out)
def conveter_aten_sum_out(
    self: Tensor,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::sum.out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sum.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sum.int)
def conveter_aten_sum_int(self: List[int], meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """NB: aten::sum.int(int[] self) -> int"""
    raise NotImplementedError("torch.ops.aten.sum.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sum.float)
def conveter_aten_sum_float(self: List[float], meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """NB: aten::sum.float(float[] self) -> float"""
    raise NotImplementedError("torch.ops.aten.sum.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sum.complex)
def conveter_aten_sum_complex(self: List[complex], meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """NB: aten::sum.complex(complex[] self) -> complex"""
    raise NotImplementedError("torch.ops.aten.sum.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sum.bool)
def conveter_aten_sum_bool(self: List[bool], meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """NB: aten::sum.bool(bool[] self) -> int"""
    raise NotImplementedError("torch.ops.aten.sum.bool ge_converter is not implemented!")
