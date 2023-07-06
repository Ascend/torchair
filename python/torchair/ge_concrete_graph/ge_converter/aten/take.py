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


@register_fx_node_ge_converter(torch.ops.aten.take.default)
def conveter_aten_take_default(
        self: Tensor,
        index: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::take(Tensor self, Tensor index) -> Tensor """
    raise NotImplementedError("torch.ops.aten.take.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.take.out)
def conveter_aten_take_out(
        self: Tensor,
        index: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.take.out ge converter is not implement!")


