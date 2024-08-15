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
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, I8, \
    U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(1024, 1024), F16(2, 2), (2, 2), (256, 256), 0),
    Support(F16(1024, 1024), F32(2, 2), (2, 2), (256, 256), 1),
    Support(F32(3, 3), F16(2, 2), (2, 2), (1, 2)),
    Support(F32(3, 3), F16(2, 2), (2, 2), (1, 2), 1),
    Support(F16(8, 2, 512, 1), F32(8, 3, 64, 1), [8, 3, 64, 1], [1024, 256, 1, 1]),
    Support(F16(96, 2, 512, 64), I32(96, 3, 512, 64), [96, 3, 512, 64], [64, 1572864, 6144, 1]),
])
@register_fx_node_ge_converter(torch.ops.aten.as_strided_scatter.default)
def conveter_aten_as_strided_scatter_default(
    self: Tensor,
    src: Tensor,
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor],
    storage_offset: Optional[Union[int, Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor"""
    storage_offset = 0 if storage_offset is None else storage_offset
    self, src = dtype_promote(self, src, target_dtype=meta_outputs.dtype)
    if isinstance(size, List):
        src_dim = len(size)
        src_stride = [1] * src_dim
        for i in range(src_dim - 1):
            src_stride[src_dim - i - 2] = src_stride[src_dim - i - 1] * size[src_dim - i - 1]
    else:
        src_dim = size.rank
        src_stride = [1] * src_dim
        for i in range(src_dim - 1):
            size_i = ge.Gather(size, src_dim - i - 1)
            src_stride[src_dim - i - 2] = ge.Mul(ge.Gather(src_stride, src_dim - i - 1), size_i)
        src_stride = ge.Pack(src_stride, N=src_dim, axis=0)
    return ge.ViewCopy(self, size, stride, storage_offset, src, size, src_stride, 0)


@register_fx_node_ge_converter(torch.ops.aten.as_strided_scatter.out)
def conveter_aten_as_strided_scatter_out(
    self: Tensor,
    src: Tensor,
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor],
    storage_offset: Optional[Union[int, Tensor]] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::as_strided_scatter.out(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.as_strided_scatter.out ge_converter is not supported!")
