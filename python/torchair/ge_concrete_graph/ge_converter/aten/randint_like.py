import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten.randint_like.default)
def conveter_aten_randint_like_default(
        self: Tensor,
        high: Union[int, Tensor],
        *,
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        memory_format: Optional[int] = None,
        meta_outputs: Any = None):
    """ NB: aten::randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.randint_like.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.randint_like.low_dtype)
def conveter_aten_randint_like_low_dtype(
        self: Tensor,
        low: Union[int, Tensor],
        high: Union[int, Tensor],
        *,
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        memory_format: Optional[int] = None,
        meta_outputs: Any = None):
    """ NB: aten::randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.randint_like.low_dtype ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.randint_like.out)
def conveter_aten_randint_like_out(
        self: Tensor,
        high: Union[int, Tensor],
        *,
        memory_format: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::randint_like.out(Tensor self, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.randint_like.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.randint_like.low_dtype_out)
def conveter_aten_randint_like_low_dtype_out(
        self: Tensor,
        low: Union[int, Tensor],
        high: Union[int, Tensor],
        *,
        memory_format: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::randint_like.low_dtype_out(Tensor self, SymInt low, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.randint_like.low_dtype_out ge converter is not implement!")


