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


@declare_supported([
    Support(F16(1, 1, 40, 128)),
    Support(U8(1, 1, 40, 128))
])
@register_fx_node_ge_converter(torch.ops.aten.alias.default)
def conveter_aten_alias_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::alias(Tensor(a) self) -> Tensor(a)"""
    shape = ge.Shape(self)
    shape = dtype_promote(shape, target_dtype=DataType.DT_INT64)
    return ge.Reshape(self, shape)
