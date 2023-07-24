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


@register_fx_node_ge_converter(torch.ops.aten.masked_fill.Scalar)
def conveter_aten_masked_fill_Scalar(
    self: Tensor, mask: Tensor, value: Union[Number, Tensor], meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor"""
    return ge.MaskedFill(self, mask, value)


@register_fx_node_ge_converter(torch.ops.aten.masked_fill.Tensor)
def conveter_aten_masked_fill_Tensor(
    self: Tensor, mask: Tensor, value: Tensor, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor"""
    return ge.MaskedFill(self, mask, value)


@register_fx_node_ge_converter(torch.ops.aten.masked_fill.Scalar_out)
def conveter_aten_masked_fill_Scalar_out(
    self: Tensor,
    mask: Tensor,
    value: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::masked_fill.Scalar_out(Tensor self, Tensor mask, Scalar value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.masked_fill.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.masked_fill.Tensor_out)
def conveter_aten_masked_fill_Tensor_out(
    self: Tensor,
    mask: Tensor,
    value: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::masked_fill.Tensor_out(Tensor self, Tensor mask, Tensor value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.masked_fill.Tensor_out ge_converter is not implemented!")
