from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, \
    F16, F64, I32, I16, I64, I8, U8, BOOL, BF16, Support


@declare_supported([
    Support(F16(16, 64, 64), F16(64, 64), F16(64, 64)),
    Support(BF16(2, 4, 4), BF16(4, 4), BF16(4, 4)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_kronecker_quant.default)
def convert_npu_kronecker_quant(
    x: Tensor,
    kronecker_p1: Tensor,
    kronecker_p2: Tensor,
    *,
    clip_ratio: Optional[float] = 1.000000,
    dst_dtype: Optional[int] = None,
    meta_outputs: Any = None
):
    y_dtype = DataType.DT_INT32
    if dst_dtype is not None and dst_dtype != torch.int32:
        raise ValueError(f"dst_dtype should be int32, "
                         f"otherwise it should be None, but got {dst_dtype}")
    if clip_ratio is None:
        clip_ratio = 1.0
    return ge.FlatQuant(x, kronecker_p1, kronecker_p2, clip_ratio=clip_ratio, dst_dtype=y_dtype)