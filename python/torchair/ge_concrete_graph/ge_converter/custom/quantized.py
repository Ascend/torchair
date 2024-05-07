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
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter, \
    torch_type_to_ge_type
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), F32(2), I32(2), 1, torch.quint8),
    Support(F16(2, 2), F32(2), I32(2), 0, torch.qint32)
])
@register_fx_node_ge_converter(torch.ops.npu.npu_quantize.default)
def conveter_npu_quantize_default(
    self: Tensor,
    scales: Tensor,
    zero_points: Optional[Tensor] = None,
    dtype: int = torch.uint8,
    axis: int = 1,
    div_mode: bool = True,
    meta_outputs: TensorSpec = None
):
    """
    NB: aten::quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points,
                                      int axis, ScalarType dtype) -> Tensor
    """
    if not div_mode:
        if dtype == torch.qint8:
            dtype = torch.int8
        return ge.AscendQuantV2(self, scales, zero_points, sqrt_mode=False, round_mode="round",
            dst_type=torch_type_to_ge_type(dtype))
    if scales.rank != 1:
        raise RuntimeError("Scales' dim should be equal to 1.")
    if axis < 0:
        axis += self.rank
    dtype_str = "torch.qint8"
    if dtype == torch.quint8:
        dtype_str = "torch.quint8"
    elif dtype == torch.qint8:
        dtype_str = "torch.qint8"
    elif dtype == torch.qint32:
        dtype_str = "torch.qint32"
    else:
        raise RuntimeError("Not supportted output dtype.")
    insert_dims = []
    for i in range(self.rank):
        if i != axis:
            insert_dims.append(i)
    if zero_points is not None:
        if zero_points.rank != 1:
            raise RuntimeError("Zero points' dim should be equal to 1.")
        zero_points = ge.Unsqueeze(zero_points, axes=insert_dims)
    scales = ge.Unsqueeze(scales, axes=insert_dims)
    return ge.Quantize(self, scales, zero_points, axis=axis, dtype=dtype_str)