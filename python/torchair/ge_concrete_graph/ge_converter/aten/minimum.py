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
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), F32(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.minimum.default)
def conveter_aten_minimum_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::minimum(Tensor self, Tensor other) -> Tensor"""
    return ge.Minimum(self, other)


@register_fx_node_ge_converter(torch.ops.aten.minimum.out)
def conveter_aten_minimum_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.minimum.out ge_converter is not implemented!")
