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


@register_fx_node_ge_converter(torch.ops.aten.scatter_.src)
def conveter_aten_scatter__src(
    self: Tensor, dim: int, index: Tensor, src: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter_.src ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter_.value)
def conveter_aten_scatter__value(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter_.value ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter_.reduce)
def conveter_aten_scatter__reduce(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    *,
    reduce: str,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter_.reduce ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter_.value_reduce)
def conveter_aten_scatter__value_reduce(
    self: Tensor,
    dim: int,
    index: Tensor,
    value: Union[Number, Tensor],
    *,
    reduce: str,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter_.value_reduce ge_converter is not implemented!")
