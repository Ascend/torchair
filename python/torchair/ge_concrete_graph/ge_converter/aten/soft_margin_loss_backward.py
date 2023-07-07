import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
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


@register_fx_node_ge_converter(torch.ops.aten.soft_margin_loss_backward.default)
def conveter_aten_soft_margin_loss_backward_default(
        grad_output: Tensor,
        self: Tensor,
        target: Tensor,
        reduction: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor """
    raise NotImplementedError("torch.ops.aten.soft_margin_loss_backward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.soft_margin_loss_backward.grad_input)
def conveter_aten_soft_margin_loss_backward_grad_input(
        grad_output: Tensor,
        self: Tensor,
        target: Tensor,
        reduction: int,
        *,
        grad_input: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::soft_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.soft_margin_loss_backward.grad_input ge converter is not implement!")


