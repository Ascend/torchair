from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote



@declare_supported(
    [
        Support(F32(2, 3, 5), [2, 0, 1]),
        Support(I8(2, 3, 5), [2, 0, 1]),
        Support(I8(2, 5, 1), [2, 0, 1]),
        Support(F32(2, 3, 4, 5), [1, 2, 3, 0]),
        Support(I32(2, 3, 1, 5), [0, 2, 1, 3]),
        Support(I32(3, 1, 1, 3), [0, 2, 1, 3]),
        Support(I32(3, 1, 1, 3), [0, 1, 2, 3]),
        Support(I32(2, 3, 1, 3, 1), [0, 2, 1, 4, 3]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.permute.default)
def conveter_aten_permute_default(
    self: Tensor, dims: List[int], meta_outputs: TensorSpec = None
):
    """NB: aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)"""
    dim_shape_self, dim_shape_meta, axis_dim, axis_unsqueeze = [], [], [], []
    for i, index in enumerate(dims):
        if index < 0:
            dims[i] = index + len(dims)
    for index, size in enumerate(self._symsize):
        if size != 1:
            dim_shape_self.append(index)
        else:
            axis_dim.append(index)
    dim_shape_meta = [i for i in dims if i not in axis_dim]
    for index, size in enumerate(meta_outputs._symsize):
        if size == 1:
            axis_unsqueeze.append(index)
    if dim_shape_self == dim_shape_meta:
        res_squeeze = ge.Squeeze(self)
        return ge.Unsqueeze(res_squeeze, axes=axis_unsqueeze)
    dims = dtype_promote(dims, target_dtype=DataType.DT_INT64)
    return ge.Transpose(self, dims)
