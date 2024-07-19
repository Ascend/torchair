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
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported([
    Support(F32(2, 2), [I64(1, 2)]),
    Support(F32(2, 2), [I64(1, 2), I64(1, 2)]),
])
@register_fx_node_ge_converter(torch.ops.aten._unsafe_index.Tensor)
def conveter_aten__unsafe_index_Tensor(
    self: Tensor, indices: List[Optional[Tensor]], meta_outputs: TensorSpec = None
):
    """NB: aten::_unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"""
    if self.dtype in [DataType.DT_BOOL, DataType.DT_UINT8]:
        raise NotImplementedError("_unsafe_index.Tensor currently not support dtype Bool or Uint8.")
    mask = [1 if indice else 0 for indice in indices]
    indices = [i for i in indices if i]
    return ge.IndexByTensor(self, indices, indices_mask=mask)
