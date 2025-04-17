from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


def get_reduction_str(reduction):
    reduction_str = ["none", "mean", "sum"]
    return reduction_str[reduction]


@declare_supported([
    Support(F32(6, 512, 44, 44), F32(6, 512, 44, 44), F32(6, 512, 44, 44), 0),
    Support(F32(6, 512, 44, 44), F32(6, 512, 44, 44), F32(6, 512, 44, 44), 1),
    Support(F32(6, 512, 44, 44), F32(6, 512, 44, 44), F32(6, 512, 44, 44), 2),
])
@register_fx_node_ge_converter(torch.ops.aten.mse_loss_backward.default)
def conveter_aten_mse_loss_backward_default(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor"""
    self, target = dtype_promote(self, target, target_dtype=meta_outputs.dtype)
    reduction_str = get_reduction_str(reduction)
    return ge.MseLossGrad(self, target, grad_output, reduction=reduction_str)


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
    raise RuntimeError("torch.ops.aten.mse_loss_backward.grad_input ge_converter is not supported!")
