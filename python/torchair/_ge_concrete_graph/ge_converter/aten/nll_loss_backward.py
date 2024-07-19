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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.utils import dtype_promote, DataType


@register_fx_node_ge_converter(torch.ops.aten.nll_loss_backward.default)
def conveter_aten_nll_loss_backward_default(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: Union[int, Tensor],
    total_weight: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight) -> Tensor"""
    if weight is None:
        weight = ge.Fill(ge.GatherV2(ge.Shape(self), 1, 0), 1.0)
    reduction_str = ['none', 'mean', 'sum']
    self, grad_output, weight, total_weight = dtype_promote(self, grad_output, weight, total_weight, target_dtype=meta_outputs.dtype)
    target_cast = dtype_promote(target, target_dtype=DataType.DT_INT32)
    grad_input = ge.NLLLossGrad(
        self, 
        grad_output, 
        target_cast, 
        weight,
        total_weight, 
        reduction=reduction_str[reduction], 
        ignore_index=ignore_index
    )
    return grad_input


@register_fx_node_ge_converter(torch.ops.aten.nll_loss_backward.grad_input)
def conveter_aten_nll_loss_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: Union[int, Tensor],
    total_weight: Tensor,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.nll_loss_backward.grad_input ge_converter is not supported!")
