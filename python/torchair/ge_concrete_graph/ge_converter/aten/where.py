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
from torchair.ge_concrete_graph.utils import dtype_promote
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(BOOL(8, 4, 10, 10), F32(8, 4, 10, 10), F32(8, 4, 10, 10)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.where.self)
def conveter_aten_where_self(
    condition: Tensor, self: Tensor, other: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor"""
    if self.desc.dtype != other.desc.dtype:
        self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
    return ge.Select(condition, self, other)


@register_fx_node_ge_converter(torch.ops.aten.where.ScalarOther)
def conveter_aten_where_ScalarOther(
    condition: Tensor,
    self: Tensor,
    other: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.where.ScalarOther ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.where.ScalarSelf)
def conveter_aten_where_ScalarSelf(
    condition: Tensor,
    self: Union[Number, Tensor],
    other: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.where.ScalarSelf ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.where.Scalar)
def conveter_aten_where_Scalar(
    condition: Tensor,
    self: Union[Number, Tensor],
    other: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.where.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.where.default)
def conveter_aten_where_default(condition: Tensor, meta_outputs: List[TensorSpec] = None):
    """NB: aten::where(Tensor condition) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.where.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.where.self_out)
def conveter_aten_where_self_out(
    condition: Tensor,
    self: Tensor,
    other: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::where.self_out(Tensor condition, Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.where.self_out ge_converter is not implemented!")
