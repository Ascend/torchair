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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), F32(2, 2), 0.),
    Support(F32(2, 2), F32(2, 2), 0.1),
    Support(F16(2, 2), F16(2, 2), 0.),
    Support(F16(2, 2), F16(2, 2), 0.1),
])
@register_fx_node_ge_converter(torch.ops.aten.threshold_backward.default)
def conveter_aten_threshold_backward_default(
    grad_output: Tensor,
    self: Tensor,
    threshold: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """ NB: aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor """
    self, grad_output = dtype_promote(self, grad_output, target_dtype=meta_outputs.dtype)

    # This scenario will be supported in the future.
    if isinstance(threshold, Tensor):
        raise RuntimeError("torch.ops.aten.threshold_backward is not implemented while threshold is Tensor!")

    # Suggest a high-performance implementation when threshold is euqal to zero.
    if threshold == 0.:
        return ge.ReluGrad(grad_output, self)
    
    return ge.ThresholdGradV2D(grad_output, self, threshold=threshold)


@register_fx_node_ge_converter(torch.ops.aten.threshold_backward.grad_input)
def conveter_aten_threshold_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    threshold: Union[Number, Tensor],
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.threshold_backward.grad_input ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
