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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote



@declare_supported(
    [
        Support(F32(2, 3, 5), [2, 0, 1]),
        Support(I8(2, 3, 5), [2, 0, 1]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.permute.default)
def conveter_aten_permute_default(
    self: Tensor, dims: List[int], meta_outputs: TensorSpec = None
):
    """NB: aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)"""
    dims = dtype_promote(dims, target_dtype=DataType.DT_INT64)
    return ge.Transpose(self, dims)
