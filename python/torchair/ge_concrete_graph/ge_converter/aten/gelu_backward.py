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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported([
    Support(F32(4, 16, 32), F32(4, 16, 32)),
    Support(F16(4, 16, 32), F16(4, 16, 32)),
])
@register_fx_node_ge_converter(torch.ops.aten.gelu_backward.default)
def conveter_aten_gelu_backward_default(
    grad_output: Tensor,
    self: Tensor,
    *,
    approximate: str = "None",
    meta_outputs: TensorSpec = None
):
    """NB: aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate="none") -> Tensor"""
    if approximate != "None":
        raise NotImplementedError("gelu_backward only supports approximate is None")
    # Due to the pattern of IR 'GeluGrad', 'unused' is fed into the input 'y'.
    unused = grad_output
    return ge.GeluGrad(grad_output, self, unused)


@register_fx_node_ge_converter(torch.ops.aten.gelu_backward.grad_input)
def conveter_aten_gelu_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    *,
    approximate: str = "None",
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gelu_backward.grad_input(Tensor grad_output, Tensor self, *, str approximate="none", Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gelu_backward.grad_input ge_converter is not implemented!")
