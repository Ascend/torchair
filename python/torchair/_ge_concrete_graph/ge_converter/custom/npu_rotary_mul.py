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
from torchair._utils.check_platform import is_not_support


@declare_supported([
    Support(F32(2, 8192, 5, 128), F32(1, 8192, 1, 128), F32(1, 8192, 1, 128)),
    Support(F16(2, 8192, 5, 128), F16(1, 8192, 1, 128), F16(1, 8192, 1, 128)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_rotary_mul.default)
def conveter_npu_rotary_mul_default(
    self: Tensor,
    r1: Tensor,
    r2: Tensor,
    rotary_mode: str = 'half',
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_rotary_mul(Tensor self, Tensor r1, Tensor r2) -> Tensor"""
    if is_not_support():
        modes = {"half": 0, "interleave": 1, "quarter": 2, "interleave-half": 3}
        if rotary_mode not in modes:
            raise NotImplementedError("rotary_mode only support half/interleave/quarter/interleave-half now!")
        mode = modes[rotary_mode]
        return ge.RotaryPositionEmbedding(self, r1, r2, mode=mode)
    else:
        return ge.RotaryMul(self, r1, r2)
