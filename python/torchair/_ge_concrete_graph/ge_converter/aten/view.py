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
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(2, 2), [4]),
        Support(F16(16), [2, 8]),
        Support(U8(16), [2, 8])
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.view.default)
def conveter_aten_view_default(
    self: Tensor, size: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)"""
    # Use dtype_promote in specific case to reduce the number of Cast operators
    if isinstance(size, list):
        size = dtype_promote(size, target_dtype=DataType.DT_INT64)
    elif size.dtype != DataType.DT_INT64 and size.dtype != DataType.DT_INT32:
        size = dtype_promote(size, target_dtype=DataType.DT_INT64)
    return ge.Reshape(self, size)


@register_fx_node_ge_converter(torch.ops.aten.view.dtype)
def conveter_aten_view_dtype(self: Tensor, dtype: int, meta_outputs: TensorSpec = None):
    """NB: aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.view.dtype ge_converter is not implemented!")
