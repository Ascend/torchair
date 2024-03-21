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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support


@declare_supported([
    Support(F32(3, 3), (2, 2), (1, 2)),
    Support(F32(3, 3), (2, 2), (1, 2), 1),
    Support(F32(3, 3), (2, 2), (1, 2), 0),
])
@register_fx_node_ge_converter(torch.ops.aten.as_strided.default)
def conveter_aten_as_strided_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor],
    storage_offset: Optional[Union[int, Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)"""
    if storage_offset is None:
        storage_offset = 0
    return ge.AsStrided(self, size, stride, storage_offset)
