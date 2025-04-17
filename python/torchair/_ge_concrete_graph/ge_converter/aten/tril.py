from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported([
    Support(F32(1, 16, 16)),
    Support(F32(1, 16, 16), diagonal=2),
    Support(F32(1, 16, 16), diagonal=-2),
    Support(F16(1, 16, 16)),
    Support(F16(1, 16, 16), diagonal=2),
    Support(F16(1, 16, 16), diagonal=-2),
    Support(BOOL(1, 16, 16)),
    Support(BOOL(1, 16, 16), diagonal=2),
    Support(BOOL(1, 16, 16), diagonal=-2),
])
@register_fx_node_ge_converter(torch.ops.aten.tril.default)
def conveter_aten_tril_default(
    self: Tensor, diagonal: int = 0, meta_outputs: TensorSpec = None
):
    """NB: aten::tril(Tensor self, int diagonal=0) -> Tensor"""
    return ge.Tril(self, diagonal=diagonal)


@register_fx_node_ge_converter(torch.ops.aten.tril.out)
def conveter_aten_tril_out(
    self: Tensor, diagonal: int = 0, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.tril.out ge_converter is not implemented!")
