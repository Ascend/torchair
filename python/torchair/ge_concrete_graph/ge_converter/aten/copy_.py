import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
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


@register_fx_node_ge_converter(torch.ops.aten.copy_.default)
def conveter_aten_copy__default(
        self: Tensor,
        src: Tensor,
        non_blocking: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.copy_.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copy_.Tensor)
def conveter_aten_copy__Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.copy_.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copy_.int)
def conveter_aten_copy__int(
        self: Tensor,
        other: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copy_.int(Tensor(a!) self, int other) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.copy_.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.copy_.float)
def conveter_aten_copy__float(
        self: Tensor,
        other: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::copy_.float(Tensor(a!) self, float other) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.copy_.float ge converter is not implement!")


