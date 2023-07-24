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
from torch import Generator, contiguous_format, inf, memory_format, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.avg_pool2d_backward.default)
def conveter_aten_avg_pool2d_backward_default(
    grad_output: Tensor,
    self: Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.avg_pool2d_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.avg_pool2d_backward.grad_input)
def conveter_aten_avg_pool2d_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    *,
    grad_input: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.avg_pool2d_backward.grad_input ge_converter is not implemented!")
