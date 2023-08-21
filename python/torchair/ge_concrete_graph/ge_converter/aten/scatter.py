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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.aten.scatter.value)
def conveter_aten_scatter_value(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.scatter.value ge_converter is not implemented!")

@declare_supported(
    [
        Support(F32(2, 2), 0, I64(2,2), F32(2, 2)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.scatter.src)
def conveter_aten_scatter_src(
    self: Tensor, dim: int, index: Tensor, src: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"""
    return ge.ScatterElements(self, index, src, axis=dim)

@register_fx_node_ge_converter(torch.ops.aten.scatter.reduce)
def conveter_aten_scatter_reduce(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    *,
    reduce: str,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.scatter.reduce ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.value_reduce)
def conveter_aten_scatter_value_reduce(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Union[Number, Tensor],
    *,
    reduce: str,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.scatter.value_reduce ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.src_out)
def conveter_aten_scatter_src_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter.src_out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter.src_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.value_out)
def conveter_aten_scatter_value_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter.value_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.reduce_out)
def conveter_aten_scatter_reduce_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    *,
    reduce: str,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter.reduce_out(Tensor self, int dim, Tensor index, Tensor src, *, str reduce, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter.reduce_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.value_reduce_out)
def conveter_aten_scatter_value_reduce_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Union[Number, Tensor],
    *,
    reduce: str,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter.value_reduce_out(Tensor self, int dim, Tensor index, Scalar value, *, str reduce, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter.value_reduce_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.dimname_src)
def conveter_aten_scatter_dimname_src(
    self: Tensor, dim: str, index: Tensor, src: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::scatter.dimname_src(Tensor self, str dim, Tensor index, Tensor src) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.scatter.dimname_src ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter.dimname_value)
def conveter_aten_scatter_dimname_value(
    self: Tensor,
    dim: str,
    index: Tensor,
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::scatter.dimname_value(Tensor self, str dim, Tensor index, Scalar value) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.scatter.dimname_value ge_converter is not implemented!")
