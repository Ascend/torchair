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


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.default)
def conveter_aten_clamp_min_default(
    self: Tensor, min: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_min(Tensor self, Scalar min) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.clamp_min.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.Tensor)
def conveter_aten_clamp_min_Tensor(self: Tensor, min: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.clamp_min.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.out)
def conveter_aten_clamp_min_out(
    self: Tensor,
    min: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_min.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_min.Tensor_out)
def conveter_aten_clamp_min_Tensor_out(
    self: Tensor, min: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_min.Tensor_out ge_converter is not implemented!")
