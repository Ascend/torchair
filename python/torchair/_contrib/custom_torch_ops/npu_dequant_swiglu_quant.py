from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload, Optional
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
    I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.npu_inference.npu_dequant_swiglu_quant.default)
def conveter_npu_dequant_swiglu_quant_default(
        x: Tensor,
        weight_scale: Tensor = None,
        activation_scale: Tensor = None,
        bias: Tensor = None,
        quant_scale: Tensor = None,
        quant_offset: Tensor = None,
        group_index: Tensor = None,
        activate_left: bool = False,
        quant_mode: int = 0,
        meta_outputs: TensorSpec = None):
    quant_mode_str = 'static'
    if quant_mode == 1:
        quant_mode_str = 'dynamic'

    return ge.DequantSwigluQuant(x, weight_scale, activation_scale, bias, quant_scale=quant_scale,
                                 quant_offset=quant_offset, group_index=group_index,
                                 activate_left=activate_left, quant_mode=quant_mode_str)