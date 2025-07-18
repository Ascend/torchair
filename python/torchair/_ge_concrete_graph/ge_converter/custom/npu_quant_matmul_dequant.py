from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter, \
    torch_type_to_ge_type
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, I64, I8, U8, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul_dequant.default)
def conveter_npu_quant_matmul_dequant_default(
    x: Tensor,
    quantized_weight: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    x_offset: Optional[Tensor] = None,
    smooth_scale: Optional[Tensor] = None,
    quant_mode: str = "pertoken",
    meta_outputs: TensorSpec = None
):
    """
    NB: aten::npu_quant_matmul_dequant(
        Tensor x, Tensor quantized_weight, Tensor weight_scale, *,
        Tensor? bias=None, Tensor? x_scale=None, Tensor? x_offset=None,
        Tensor? smooth_scale=None,
        str? quant_mode='pertoken') -> Tensor
    """
    return ge.QuantMatmulDequant(
        x, quantized_weight, weight_scale,
        bias=bias, x_scale=x_scale, x_offset=x_offset, smooth_scale=smooth_scale, x_quant_mode=quant_mode)