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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import F32, F16, BF16, I32, Support


@declare_supported(
    [
        Support([I32(2, 2, 2), I32(2, 3), I32(2, 3)], 1),
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], 1.),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], 1.),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], 1.),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_add.Scalar)
def conveter_aten__foreach_add_Scalar(
    self: List[Tensor], scalar: Union[Number, Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_add.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]"""
    return ge.ForeachAddScalar(self, scalar)


@register_fx_node_ge_converter(torch.ops.aten._foreach_add.List)
def conveter_aten__foreach_add_List(
    self: List[Tensor],
    other: List[Tensor],
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_add.List(Tensor[] self, Tensor[] other, *, Scalar alpha=1) -> Tensor[]"""
    return ge.ForeachAddList(self, other, alpha)


@declare_supported(
    [
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], [1., 2., 3.]),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], [1., 2., 3.]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_add.ScalarList)
def conveter_aten__foreach_add_ScalarList(
    self: List[Tensor], scalars: Union[List[Number], Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_add.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]"""
    return ge.ForeachAddScalarList(self, scalars)


@register_fx_node_ge_converter(torch.ops.aten._foreach_add.Scalar_out)
def conveter_aten__foreach_add_Scalar_out(
    self: List[Tensor],
    scalar: Union[Number, Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_add.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_add.Scalar_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_add.List_out)
def conveter_aten__foreach_add_List_out(
    self: List[Tensor],
    other: List[Tensor],
    *,
    alpha: Union[Number, Tensor] = 1,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_add.List_out(Tensor[] self, Tensor[] other, *, Scalar alpha=1, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_add.List_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_add.ScalarList_out)
def conveter_aten__foreach_add_ScalarList_out(
    self: List[Tensor],
    scalars: Union[List[Number], Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_add.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_add.ScalarList_out ge_converter is not supported!")
