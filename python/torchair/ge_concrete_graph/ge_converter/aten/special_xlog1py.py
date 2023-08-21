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


@register_fx_node_ge_converter(torch.ops.aten.special_xlog1py.default)
def conveter_aten_special_xlog1py_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::special_xlog1py(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_xlog1py.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_xlog1py.other_scalar)
def conveter_aten_special_xlog1py_other_scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::special_xlog1py.other_scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_xlog1py.other_scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_xlog1py.self_scalar)
def conveter_aten_special_xlog1py_self_scalar(
    self: Union[Number, Tensor], other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::special_xlog1py.self_scalar(Scalar self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_xlog1py.self_scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_xlog1py.out)
def conveter_aten_special_xlog1py_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::special_xlog1py.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_xlog1py.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_xlog1py.self_scalar_out)
def conveter_aten_special_xlog1py_self_scalar_out(
    self: Union[Number, Tensor],
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::special_xlog1py.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_xlog1py.self_scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_xlog1py.other_scalar_out)
def conveter_aten_special_xlog1py_other_scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::special_xlog1py.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_xlog1py.other_scalar_out ge_converter is not implemented!")
