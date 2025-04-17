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


@declare_supported([
    Support(F32(32, 768), F32(32, 768)),
    Support(F16(32, 768), F16(32, 768)),
])
@register_fx_node_ge_converter(torch.ops.aten.tanh_backward.default)
def conveter_aten_tanh_backward_default(
    grad_output: Tensor, output: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor"""
    return ge.TanhGrad(output, grad_output)


@register_fx_node_ge_converter(torch.ops.aten.tanh_backward.grad_input)
def conveter_aten_tanh_backward_grad_input(
    grad_output: Tensor,
    output: Tensor,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.tanh_backward.grad_input ge_converter is not supported!")
