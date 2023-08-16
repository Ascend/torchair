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
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(64, 4, 9), F32(64, 8, 9), 1),
        Support(F32(64, 4, 4), F32(64, 4, 8), 2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.glu_backward.default)
def conveter_aten_glu_backward_default(
    grad_output: Tensor, self: Tensor, dim: int, meta_outputs: TensorSpec = None
):
    """NB: aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor"""
    return ge.GLUGrad(grad_output, self, dim=dim)


@register_fx_node_ge_converter(torch.ops.aten.glu_backward.grad_input)
def conveter_aten_glu_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    dim: int,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.glu_backward.grad_input ge_converter is not implemented!")
