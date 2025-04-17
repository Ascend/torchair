from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F16(16, 128), F16(32, 128), F16(32,)),

    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_linear.default)
def conveter_npu_npu_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    """
    if input.dtype == DataType.DT_INT8 or weight.dtype == DataType.DT_INT8:
        raise RuntimeError("torch.ops.aten.npu_linear.default ge_converter is not support int8 dtype!")
    return ge.MatMul(input, weight, bias=bias, transpose_x1=False, transpose_x2=True)
