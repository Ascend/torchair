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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(8, 1, 128), F32(128)),
    Support(F16(8, 1, 128), F16(128)),
    # support setting epsilon
    Support(F16(8, 1, 128), F16(128), float(0.01)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_add_rms_norm.default)
def conveter_npu_add_rms_norm_default(
    x1: Tensor,
    x2: Tensor,
    gamma: Tensor,
    epsilon: float = 1e-6,
    meta_outputs: List[TensorSpec] = None
):
    """NB: npu::npu_add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor, Tensor)"""
    return ge.AddRmsNorm(x1, x2, gamma, epsilon=epsilon)
