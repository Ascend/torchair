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
    Support(F32(1, 1, 40, 2, 64), 3)
])
@register_fx_node_ge_converter(torch.ops.aten.unbind.int)
def conveter_aten_unbind_int(self: Tensor, dim: int = 0, meta_outputs: List[TensorSpec] = None):
    """NB: aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]"""
    num = len(meta_outputs)
    return ge.Unpack(self, num=num, axis=dim)


@register_fx_node_ge_converter(torch.ops.aten.unbind.Dimname)
def conveter_aten_unbind_Dimname(self: Tensor, dim: str, meta_outputs: List[TensorSpec] = None):
    """NB: aten::unbind.Dimname(Tensor(a -> *) self, str dim) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.unbind.Dimname ge_converter is not implemented!")
