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


@register_fx_node_ge_converter(torch.ops.aten.mse_loss_backward.default)
def conveter_aten_mse_loss_backward_default(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.mse_loss_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.mse_loss_backward.grad_input)
def conveter_aten_mse_loss_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.mse_loss_backward.grad_input ge_converter is not implemented!")
