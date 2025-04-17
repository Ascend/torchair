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
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(2, 2), F32(2, 2)),
        Support(F32(2, 2), F32(2, 1)),
        Support(F32(2, 2), F16(2, 1)),
        Support(F32(2, 2), F16(2, 2), alpha=2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.rsub.Tensor)
def conveter_aten_rsub_Tensor(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    if not isinstance(alpha, Tensor) and alpha == 1:
        # just for better permance
        self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
        return ge.Sub(other, self)
    else:
        self, other, alpha = dtype_promote(self, other, alpha, target_dtype=meta_outputs.dtype)
        self_mul = ge.Mul(self, alpha)
        return ge.Sub(other, self_mul)



@declare_supported(
    [
        Support(F32(2, 2), 2.0),
        Support(F32(2, 2), 2),
        Support(F32(2, 2), 2, alpha=2.0),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.rsub.Scalar)
def conveter_aten_rsub_Scalar(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"""
    self, other, alpha = dtype_promote(self, other, alpha, target_dtype=meta_outputs.dtype)
    return ge.Sxpy(other, self, alpha=alpha)


@register_fx_node_ge_converter(torch.ops.aten.rsub.Tensor_out)
def conveter_aten_rsub_Tensor_out(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rsub.Tensor_out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rsub.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rsub.Scalar_out)
def conveter_aten_rsub_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    alpha: Union[Number, Tensor] = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rsub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rsub.Scalar_out ge_converter is not implemented!")
