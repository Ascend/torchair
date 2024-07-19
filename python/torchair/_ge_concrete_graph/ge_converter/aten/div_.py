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


@register_fx_node_ge_converter(torch.ops.aten.div_.Tensor)
def conveter_aten_div__Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.div_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.div_.Scalar)
def conveter_aten_div__Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.div_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.div_.Tensor_mode)
def conveter_aten_div__Tensor_mode(
    self: Tensor,
    other: Tensor,
    *,
    rounding_mode: Optional[str],
    meta_outputs: TensorSpec = None
):
    """NB: aten::div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.div_.Tensor_mode ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.div_.Scalar_mode)
def conveter_aten_div__Scalar_mode(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    rounding_mode: Optional[str],
    meta_outputs: TensorSpec = None
):
    """NB: aten::div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.div_.Scalar_mode ge_converter is not implemented!")
