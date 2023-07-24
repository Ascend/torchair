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


@register_fx_node_ge_converter(torch.ops.aten.rsub.Tensor)
def conveter_aten_rsub_Tensor(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    mul_self = self
    if alpha != 1:
        mul_self = ge.Mul(self, alpha)
    return ge.Sub(mul_self, other)


@register_fx_node_ge_converter(torch.ops.aten.rsub.Scalar)
def conveter_aten_rsub_Scalar(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"""
    mul_self = self
    if alpha != 1:
        mul_self = ge.Mul(self, alpha)
    return ge.Sub(mul_self, other)


@register_fx_node_ge_converter(torch.ops.aten.rsub.Tensor_out)
def conveter_aten_rsub_Tensor_out(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::rsub.Tensor_out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rsub.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rsub.Scalar_out)
def conveter_aten_rsub_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::rsub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rsub.Scalar_out ge_converter is not implemented!")
