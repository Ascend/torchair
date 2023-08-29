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
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), 1),
])
@register_fx_node_ge_converter(torch.ops.aten.fill.Scalar)
def conveter_aten_fill_Scalar(
    self: Tensor, value: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::fill.Scalar(Tensor self, Scalar value) -> Tensor"""
    dims = ge.Shape(self)
    return ge.Fill(dims, ge.Cast(value, dst_type=self.dtype))

@register_fx_node_ge_converter(torch.ops.aten.fill.Scalar_out)
def conveter_aten_fill_Scalar_out(
    self: Tensor,
    value: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::fill.Scalar_out(Tensor self, Scalar value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fill.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.fill.Tensor)
def conveter_aten_fill_Tensor(self: Tensor, value: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::fill.Tensor(Tensor self, Tensor value) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.fill.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.fill.Tensor_out)
def conveter_aten_fill_Tensor_out(
    self: Tensor, value: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::fill.Tensor_out(Tensor self, Tensor value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.fill.Tensor_out ge_converter is not implemented!")
