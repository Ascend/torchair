from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.log_sigmoid_backward.default)
def conveter_aten_log_sigmoid_backward_default(
    grad_output: Tensor, self: Tensor, buffer: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.log_sigmoid_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log_sigmoid_backward.grad_input)
def conveter_aten_log_sigmoid_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    buffer: Tensor,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.log_sigmoid_backward.grad_input ge_converter is not implemented!")
