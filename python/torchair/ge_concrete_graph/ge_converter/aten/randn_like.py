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


@register_fx_node_ge_converter(torch.ops.aten.randn_like.default)
def conveter_aten_randn_like_default(
    self: Tensor,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.randn_like.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randn_like.out)
def conveter_aten_randn_like_out(
    self: Tensor,
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randn_like.out ge_converter is not implemented!")
