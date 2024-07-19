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

import sys
import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, I32, I64, I8, U8, Support
from torchair.ge._ge_graph import DataType


@declare_supported([
    Support(I8(24, 4096, 128), I32(24), BF16(24, 1, 128), BF16(1, 1, 128), BF16(1, 1, 128),
            axis=-2, quant_axis=-1, reduce="update"),
    Support(I8(24, 4096, 128), I32(24), BF16(24, 1, 128), BF16(1, 1, 128),
            axis=-2, quant_axis=-1, reduce="update"),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_quant_scatter.default)
def conveter_npu_quant_scatter_default(
    self: Tensor,
    indices: Tensor,
    updates: Tensor,
    quant_scales: Tensor,
    quant_zero_points: Optional[Tensor] = None,
    axis: int = 0,
    quant_axis: int = 1,
    reduce: str = 'update',
    meta_outputs: TensorSpec = None
):
    """
    NB: aten::npu_quant_scatter(Tensor self, Tensor indices, Tensor updates, Tensor quant_scales,
                                Tensor? quant_zero_points=None, int axis=0, int quant_axis=1,
                                str reduce='update') -> Tensor
    """
    """
    Warning: kernel [npu_quant_scatter] is a out-of-place op, but it is supported by another in-place op 
             cann.QuantUpdateScatter. This current usage may cause the input to be changed unexpectedly, and the caller 
             needs to pay attention to this feature.
    """

    copy = ge.TensorMove(self)
    return ge.QuantUpdateScatter(copy, indices, updates, quant_scales, quant_zero_points, reduce=reduce, axis=axis,
                                 quant_axis=quant_axis)
