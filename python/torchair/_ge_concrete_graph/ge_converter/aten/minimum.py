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


@declare_supported([
    Support(F32(2, 2), F32(2, 2)),
    Support(F32(32), F16(32)),
    Support(F16(32), I8(32)),
    Support(I8(32), BOOL(32)),
    Support(F16(32), BOOL(32)),
    Support(I64(32), I8(32)),
    Support(U8(32), I8(32)),
    Support(U8(32), U8(32)),
    Support(I8(32), I16(32)),
    Support(BOOL(32), BOOL(32)),
])
@register_fx_node_ge_converter(torch.ops.aten.minimum.default)
def conveter_aten_minimum_default(
    self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::minimum(Tensor self, Tensor other) -> Tensor"""
    if meta_outputs.dtype == DataType.DT_BOOL:
        return ge.Mul(self, other)
    if meta_outputs.dtype in [DataType.DT_UINT8, DataType.DT_INT16]:
        self, other = dtype_promote(self, other, target_dtype=DataType.DT_INT32)
        outputs = ge.Minimum(self, other)
        return dtype_promote(outputs, target_dtype=meta_outputs.dtype)
    self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Minimum(self, other)


@register_fx_node_ge_converter(torch.ops.aten.minimum.out)
def conveter_aten_minimum_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.minimum.out ge_converter is not supported!")
