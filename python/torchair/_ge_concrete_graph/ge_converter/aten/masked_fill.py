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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    T, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(16), BOOL(16), 0.0),
    Support(F32(16, 16), BOOL(16, 16), 0),
    Support(F16(16, 16), BOOL(16, 16), 0),
])
@register_fx_node_ge_converter(torch.ops.aten.masked_fill.Scalar)
def conveter_aten_masked_fill_Scalar(
    self: Tensor, mask: Tensor, value: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor"""
    value = dtype_promote(value, target_dtype=meta_outputs.dtype)
    return ge.MaskedFill(self, mask, value)


@declare_supported([
    Support(F32(16), BOOL(16), T(0, dtype=torch.float32)),
    Support(F32(16, 16), BOOL(16, 16), T(0, dtype=torch.int64)),
    Support(F16(16, 16), BOOL(16, 16), T(0, dtype=torch.float16)),
])
@register_fx_node_ge_converter(torch.ops.aten.masked_fill.Tensor)
def conveter_aten_masked_fill_Tensor(
        self: Tensor, mask: Tensor, value: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor"""
    value = dtype_promote(value, target_dtype=meta_outputs.dtype)
    return ge.MaskedFill(self, mask, value)


@register_fx_node_ge_converter(torch.ops.aten.masked_fill.Scalar_out)
def conveter_aten_masked_fill_Scalar_out(
    self: Tensor,
    mask: Tensor,
    value: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::masked_fill.Scalar_out(Tensor self, Tensor mask, Scalar value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.masked_fill.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.masked_fill.Tensor_out)
def conveter_aten_masked_fill_Tensor_out(
    self: Tensor,
    mask: Tensor,
    value: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::masked_fill.Tensor_out(Tensor self, Tensor mask, Tensor value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.masked_fill.Tensor_out ge_converter is not implemented!")
