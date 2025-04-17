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


@declare_supported( # 典型shape
    [
        Support(F16(96, 2, 2, 32, 32), F16(2, 32, 32), F16(2, 32, 32),
            scale_value=1, inner_precision_mode=0),
        Support(F32(96, 2, 2, 32, 32), F32(2, 32, 32), F16(2, 32, 32),
        scale_value=1, inner_precision_mode=0),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_masked_softmax_with_rel_pos_bias.default)
def convert_npu_npu_masked_softmax_with_rel_pos_bias(
    x: Tensor,
    atten_mask: Optional[Tensor],
    relative_pos_bias: Tensor,
    scale_value: float = 1.0,
    inner_precision_mode: int = 0,
    meta_outputs: TensorSpec = None
):
    return ge.MaskedSoftmaxWithRelPosBias(x, atten_mask, relative_pos_bias, scale_value=scale_value,
        inner_precision_mode=inner_precision_mode)
