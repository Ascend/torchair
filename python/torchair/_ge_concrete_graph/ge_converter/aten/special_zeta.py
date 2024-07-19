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


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.default)
def conveter_aten_special_zeta_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_zeta.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.other_scalar)
def conveter_aten_special_zeta_other_scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_zeta.other_scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.self_scalar)
def conveter_aten_special_zeta_self_scalar(
    self: Union[Number, Tensor], other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.special_zeta.self_scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.out)
def conveter_aten_special_zeta_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_zeta.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.self_scalar_out)
def conveter_aten_special_zeta_self_scalar_out(
    self: Union[Number, Tensor],
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_zeta.self_scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.special_zeta.other_scalar_out)
def conveter_aten_special_zeta_other_scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::special_zeta.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.special_zeta.other_scalar_out ge_converter is not implemented!")
