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
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(4, 2048, 5120), F32(4, 2048, 5120), F32(5120,), F32(4, 2048, 1)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_rms_norm_backward.default)
def conveter_npu_rms_norm_backward_default(
    grad: Tensor,
    self: Tensor,
    gamma: Tensor,
    rstd: Tensor,
    meta_outputs: List[TensorSpec] = None
):
    """NB: npu::npu_rms_norm_backward(Tensor dy, Tensor self, Tensor gamma, Tensor rstd) -> (Tensor, Tensor)"""
    dx, dgamma = ge.RmsNormGrad(grad, self, rstd, gamma)
    return dx, dgamma
