from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(2, 115, 32, 32), 1, 2, 3, False, F32(2, 115, 32, 32)),
        Support(F32(2, 32, 32), 1.5, 1.2, 1.4, False, F32(2, 32, 32)),
        Support(F32(2, 32, 32), 1.5, 1.2, 1.4, True, F32(2, 32, 32)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.elu_backward.default)
def conveter_aten_elu_backward_default(
    grad_output: Tensor,
    alpha: Union[Number, Tensor],
    scale: Union[Number, Tensor],
    input_scale: Union[Number, Tensor],
    is_result: bool,
    self_or_result: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor"""
    if isinstance(alpha, Tensor) or isinstance(scale, Tensor) or isinstance(input_scale, Tensor):
        raise RuntimeError("ge.EluGradV2 is not support when alpha or scale or input_scale is Tensor")
    if is_result and alpha < 0:
        raise RuntimeError("torch.ops.aten.elu_backward.default is triggered with a negative slope which is not "
                           "supported")
    return ge.EluGradV2(grad_output, self_or_result, alpha=alpha,
                        scale=scale, input_scale=input_scale, is_result=is_result)


@register_fx_node_ge_converter(torch.ops.aten.elu_backward.grad_input)
def conveter_aten_elu_backward_grad_input(
    grad_output: Tensor,
    alpha: Union[Number, Tensor],
    scale: Union[Number, Tensor],
    input_scale: Union[Number, Tensor],
    is_result: bool,
    self_or_result: Tensor,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::elu_backward.grad_input(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.elu_backward.grad_input ge_converter is not supported!")
