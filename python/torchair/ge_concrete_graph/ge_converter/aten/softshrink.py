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


@register_fx_node_ge_converter(torch.ops.aten.softshrink.default)
def conveter_aten_softshrink_default(
        self: Tensor,
        lambd: Union[Number, Tensor] = 0.5,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor """
    raise NotImplementedError("torch.ops.aten.softshrink.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.softshrink.out)
def conveter_aten_softshrink_out(
        self: Tensor,
        lambd: Union[Number, Tensor] = 0.5,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.softshrink.out ge converter is not implement!")


