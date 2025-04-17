from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, BF16, F32, F16, F64, I32, I16, \
    I64, I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote, DataType


@declare_supported([
    Support(F32(2, 2), dtype=torch.bool),
    Support(F16(16), dtype=torch.int32),
    Support(F32(8), dtype=torch.float16),
    Support(F16(4, 6), dtype=torch.float32),
    Support(F16(2, 1, 3, 4), dtype=torch.float16),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_dtype_cast.default)
def conveter_npu_dtype_cast_default(
    self: Tensor,
    dtype: int,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_dtype_cast(Tensor self, ScalarType dtype) -> Tensor"""
    return ge.Cast(self, dst_type=torch_type_to_ge_type(dtype))