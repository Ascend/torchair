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


@register_fx_node_ge_converter(torch.ops.aten.std.default)
def conveter_aten_std_default(
    self: Tensor, unbiased: bool = True, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::std(Tensor self, bool unbiased=True) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.std.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.dim)
def conveter_aten_std_dim(
    self: Tensor,
    dim: Optional[List[int]],
    unbiased: bool = True,
    keepdim: bool = False,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.std.dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.correction)
def conveter_aten_std_correction(
    self: Tensor,
    dim: Optional[List[int]] = None,
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.std.correction ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.names_dim)
def conveter_aten_std_names_dim(
    self: Tensor,
    dim: List[str],
    unbiased: bool = True,
    keepdim: bool = False,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::std.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.std.names_dim ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.names_out)
def conveter_aten_std_names_out(
    self: Tensor,
    dim: List[str],
    unbiased: bool = True,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::std.names_out(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.std.names_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.out)
def conveter_aten_std_out(
    self: Tensor,
    dim: Optional[List[int]],
    unbiased: bool = True,
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::std.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.std.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.correction_out)
def conveter_aten_std_correction_out(
    self: Tensor,
    dim: Optional[List[int]] = None,
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::std.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.std.correction_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.correction_names)
def conveter_aten_std_correction_names(
    self: Tensor,
    dim: List[str],
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::std.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.std.correction_names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.std.correction_names_out)
def conveter_aten_std_correction_names_out(
    self: Tensor,
    dim: List[str],
    *,
    correction: Optional[Union[Number, Tensor]] = None,
    keepdim: bool = False,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::std.correction_names_out(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.std.correction_names_out ge_converter is not implemented!")
