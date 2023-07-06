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


@register_fx_node_ge_converter(torch.ops.aten.sigmoid.default)
def conveter_aten_sigmoid_default(
        self: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::sigmoid(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.sigmoid.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sigmoid.out)
def conveter_aten_sigmoid_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.sigmoid.out ge converter is not implement!")


