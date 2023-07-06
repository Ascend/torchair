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


@register_fx_node_ge_converter(torch.ops.aten.softshrink_backward.default)
def conveter_aten_softshrink_backward_default(
        grad_output: Tensor,
        self: Tensor,
        lambd: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor """
    raise NotImplementedError("torch.ops.aten.softshrink_backward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.softshrink_backward.grad_input)
def conveter_aten_softshrink_backward_grad_input(
        grad_output: Tensor,
        self: Tensor,
        lambd: Union[Number, Tensor],
        *,
        grad_input: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::softshrink_backward.grad_input(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.softshrink_backward.grad_input ge converter is not implement!")


