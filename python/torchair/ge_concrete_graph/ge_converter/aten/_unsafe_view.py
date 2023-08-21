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


@register_fx_node_ge_converter(torch.ops.aten._unsafe_view.default)
def conveter_aten__unsafe_view_default(
    self: Tensor, size: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::_unsafe_view(Tensor self, SymInt[] size) -> Tensor"""
    return ge.Reshape(self, size)


@register_fx_node_ge_converter(torch.ops.aten._unsafe_view.out)
def conveter_aten__unsafe_view_out(
    self: Tensor,
    size: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_unsafe_view.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._unsafe_view.out ge_converter is not implemented!")
