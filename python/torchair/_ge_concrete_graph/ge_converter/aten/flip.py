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
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(3, 4, 5), [0]),
    Support(I32(3, 4), [1]),
    Support(I32(3, 4), [0, 1]),
])
@register_fx_node_ge_converter(torch.ops.aten.flip.default)
def conveter_aten_flip_default(self: Tensor, dims: List[int], meta_outputs: TensorSpec = None):
    """NB: aten::flip(Tensor self, int[] dims) -> Tensor"""
    return ge.ReverseV2(self, dims)


@register_fx_node_ge_converter(torch.ops.aten.flip.out)
def conveter_aten_flip_out(
    self: Tensor, dims: List[int], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::flip.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.flip.out ge_converter is not implemented!")
