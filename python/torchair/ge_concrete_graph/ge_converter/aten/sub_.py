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


@register_fx_node_ge_converter(torch.ops.aten.sub_.Tensor)
def conveter_aten_sub__Tensor(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sub_.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sub_.Scalar)
def conveter_aten_sub__Scalar(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sub_.Scalar ge_converter is not implemented!")
