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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type, DataType
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported([
    Support(F32(3, 4), F32(3, 4), 0, torch.float32),
    Support(F32(3, 4), F32(3, 4), 1, torch.float32),
])
@register_fx_node_ge_converter(torch.ops.aten._log_softmax_backward_data.default)
def conveter_aten__log_softmax_backward_data_default(
    grad_output: Tensor,
    output: Tensor,
    dim: int,
    input_dtype: int,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"""
    input_ge_type = torch_type_to_ge_type(input_dtype)
    half_to_float = grad_output.dtype != input_ge_type
    if half_to_float:
        if grad_output.dtype != DataType.DT_FLOAT or \
            input_ge_type != DataType.DT_FLOAT16:
                raise NotImplementedError("expected input and grad types to match,",
                        " or input to be at::Half and grad to be at::Float")
    return ge.LogSoftmaxGrad(grad_output, output, axis=[dim])


@register_fx_node_ge_converter(torch.ops.aten._log_softmax_backward_data.out)
def conveter_aten__log_softmax_backward_data_out(
    grad_output: Tensor,
    output: Tensor,
    dim: int,
    input_dtype: int,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten._log_softmax_backward_data.out ge_converter is not supported!")
