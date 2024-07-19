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


@register_fx_node_ge_converter(torch.ops.aten.index_fill.int_Tensor)
def conveter_aten_index_fill_int_Tensor(
    self: Tensor, dim: int, index: Tensor, value: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.index_fill.int_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill.int_Scalar)
def conveter_aten_index_fill_int_Scalar(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.index_fill.int_Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill.Dimname_Scalar)
def conveter_aten_index_fill_Dimname_Scalar(
    self: Tensor,
    dim: str,
    index: Tensor,
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_fill.Dimname_Scalar(Tensor self, str dim, Tensor index, Scalar value) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.index_fill.Dimname_Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill.Dimname_Tensor)
def conveter_aten_index_fill_Dimname_Tensor(
    self: Tensor, dim: str, index: Tensor, value: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_fill.Dimname_Tensor(Tensor self, str dim, Tensor index, Tensor value) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.index_fill.Dimname_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill.int_Scalar_out)
def conveter_aten_index_fill_int_Scalar_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_fill.int_Scalar_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_fill.int_Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_fill.int_Tensor_out)
def conveter_aten_index_fill_int_Tensor_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_fill.int_Tensor_out(Tensor self, int dim, Tensor index, Tensor value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_fill.int_Tensor_out ge_converter is not implemented!")
