from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 3), F32(2, 3)),
    Support(F16(2, 3), F16(2, 3))
])
@register_fx_node_ge_converter(torch.ops.aten.less.Tensor)
def conveter_aten_less_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::less.Tensor(Tensor self, Tensor other) -> Tensor"""
    if self.dtype != other.dtype:
        other = dtype_promote(other, target_dtype=self.dtype)
    return ge.Less(self, other)


@register_fx_node_ge_converter(torch.ops.aten.less.Scalar)
def conveter_aten_less_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::less.Scalar(Tensor self, Scalar other) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.less.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.less.Scalar_out)
def conveter_aten_less_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::less.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.less.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.less.Tensor_out)
def conveter_aten_less_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::less.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.less.Tensor_out ge_converter is not implemented!")
