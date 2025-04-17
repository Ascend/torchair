from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I64, Support
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), F32(2, 2)),
    Support(I32(2, 2), F32(2, 2)),
    Support(F64(2, 2), I64(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.logical_and.default)
def conveter_aten_logical_and_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::logical_and(Tensor self, Tensor other) -> Tensor"""
    self, other = dtype_promote(self, other, target_dtype=DataType.DT_BOOL)
    return ge.LogicalAnd(self, other)


@register_fx_node_ge_converter(torch.ops.aten.logical_and.out)
def conveter_aten_logical_and_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logical_and.out ge_converter is not implemented!")
