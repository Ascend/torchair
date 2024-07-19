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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(4, 1, 1664), [1408, 128, 128], 2),
    Support(U8(4, 1, 1664), [1408, 128, 128], 2)
])
@register_fx_node_ge_converter(torch.ops.aten.split_with_sizes.default)
def conveter_aten_split_with_sizes_default(
    self: Tensor,
    split_sizes: Union[List[int], Tensor],
    dim: int = 0,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]"""
    split_sizes = dtype_promote(split_sizes, target_dtype=DataType.DT_INT64)
    if dim > 2147483647:
        dim = dtype_promote(dim, target_dtype=DataType.DT_INT64)
    else:
        dim = dtype_promote(dim, target_dtype=DataType.DT_INT32)
    return ge.SplitV(self, split_sizes, dim, num_split=len(meta_outputs))