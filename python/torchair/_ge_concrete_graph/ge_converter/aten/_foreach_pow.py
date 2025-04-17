from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import F32, F16, BF16, I32, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], [F32(2, 2, 2), F32(2, 3), F32(2, 3)]),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], [F16(2, 2, 2), F16(2, 3), F16(2, 3)]),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], [BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.List)
def conveter_aten__foreach_pow_List(
    self: List[Tensor], exponent: List[Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_pow.List(Tensor[] self, Tensor[] exponent) -> Tensor[]"""
    return ge.ForeachPowList(self, exponent)


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], 1.),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], 1.),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], 1.),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.Scalar)
def conveter_aten__foreach_pow_Scalar(
    self: List[Tensor], exponent: Union[Number, Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_pow.Scalar(Tensor[] self, Scalar exponent) -> Tensor[]"""
    if len(self) > 0:
        if self[0].dtype == DataType.DT_BF16:
            exponent = dtype_promote(exponent, target_dtype=DataType.DT_FLOAT)
        else:
            exponent = dtype_promote(exponent, target_dtype=self[0].dtype)
    return ge.ForeachPowScalar(self, exponent)


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], [1., 1., 1.]),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], [1., 1., 1.]),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], [1., 1., 1.]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.ScalarList)
def conveter_aten__foreach_pow_ScalarList(
    self: List[Tensor], exponent: Union[List[Number], Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_pow.ScalarList(Tensor[] self, Scalar[] exponent) -> Tensor[]"""
    return ge.ForeachPowScalarList(self, exponent)


@declare_supported(
    [
        Support(1., [F32(2, 2, 2), F32(2, 3), F32(2, 3)]),
        Support(1., [F16(2, 2, 2), F16(2, 3), F16(2, 3)]),
        Support(1., [BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.ScalarAndTensor)
def conveter_aten__foreach_pow_ScalarAndTensor(
    self: Union[Number, Tensor], exponent: List[Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_pow.ScalarAndTensor(Scalar self, Tensor[] exponent) -> Tensor[]"""
    if isinstance(self, float):
        self = ge.Fill([1], ge.Cast(self, dst_type=DataType.DT_FLOAT))
    return ge.ForeachPowScalarAndTensor(self, exponent)


@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.List_out)
def conveter_aten__foreach_pow_List_out(
    self: List[Tensor],
    exponent: List[Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_pow.List_out(Tensor[] self, Tensor[] exponent, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_pow.List_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.Scalar_out)
def conveter_aten__foreach_pow_Scalar_out(
    self: List[Tensor],
    exponent: Union[Number, Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_pow.Scalar_out(Tensor[] self, Scalar exponent, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_pow.Scalar_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.ScalarList_out)
def conveter_aten__foreach_pow_ScalarList_out(
    self: List[Tensor],
    exponent: Union[List[Number], Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_pow.ScalarList_out(Tensor[] self, Scalar[] exponent, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_pow.ScalarList_out ge_converter is not supported!")
