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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(2, 2), dim=0),
        Support(F32(2, 2), dim=1),
        Support(F32(2, 2), dim=2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.unsqueeze.default)
def conveter_aten_unsqueeze_default(self: Tensor, dim: int, meta_outputs: TensorSpec = None):
    """NB: aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)"""
    return ge.Unsqueeze(self, axes=[dim]);
