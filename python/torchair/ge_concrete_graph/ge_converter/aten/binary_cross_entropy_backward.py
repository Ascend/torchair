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


@register_fx_node_ge_converter(torch.ops.aten.binary_cross_entropy_backward.default)
def conveter_aten_binary_cross_entropy_backward_default(
        grad_output: Tensor,
        self: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        reduction: int = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.binary_cross_entropy_backward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.binary_cross_entropy_backward.grad_input)
def conveter_aten_binary_cross_entropy_backward_grad_input(
        grad_output: Tensor,
        self: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        reduction: int = 1,
        *,
        grad_input: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=1, *, Tensor(a!) grad_input) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.binary_cross_entropy_backward.grad_input ge converter is not implement!")


