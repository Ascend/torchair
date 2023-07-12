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


@register_fx_node_ge_converter(torch.ops.aten.roll.default)
def conveter_aten_roll_default(
        self: Tensor,
        shifts: Union[List[int], Tensor],
        dims: List[int] = [],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::roll(Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor """
    raise NotImplementedError("torch.ops.aten.roll.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.roll.out)
def conveter_aten_roll_out(
        self: Tensor,
        shifts: Union[List[int], Tensor],
        dims: List[int] = [],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::roll.out(Tensor self, SymInt[1] shifts, int[1] dims=[], *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.roll.out ge converter is not implement!")


