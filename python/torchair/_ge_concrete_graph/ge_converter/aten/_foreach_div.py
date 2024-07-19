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


@register_fx_node_ge_converter(torch.ops.aten._foreach_div.Scalar)
def conveter_aten__foreach_div_Scalar(
    self: List[Tensor], scalar: Union[Number, Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_div.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten._foreach_div.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_div.List)
def conveter_aten__foreach_div_List(
    self: List[Tensor], other: List[Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_div.List(Tensor[] self, Tensor[] other) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten._foreach_div.List ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_div.ScalarList)
def conveter_aten__foreach_div_ScalarList(
    self: List[Tensor], scalars: Union[List[Number], Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_div.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten._foreach_div.ScalarList ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_div.Scalar_out)
def conveter_aten__foreach_div_Scalar_out(
    self: List[Tensor],
    scalar: Union[Number, Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_div.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_div.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_div.List_out)
def conveter_aten__foreach_div_List_out(
    self: List[Tensor],
    other: List[Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_div.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_div.List_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_div.ScalarList_out)
def conveter_aten__foreach_div_ScalarList_out(
    self: List[Tensor],
    scalars: Union[List[Number], Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_div.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_div.ScalarList_out ge_converter is not implemented!")
