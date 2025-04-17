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
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], 1.),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], 1.),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], 1.),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_div.Scalar)
def conveter_aten__foreach_div_Scalar(
    self: List[Tensor], scalar: Union[Number, Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_div.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]"""
    if len(self) > 0:
        if self[0].dtype == DataType.DT_BF16:
            scalar = dtype_promote(scalar, target_dtype=DataType.DT_FLOAT)
        else:
            scalar = dtype_promote(scalar, target_dtype=self[0].dtype)
    return ge.ForeachDivScalar(self, scalar)


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], [F32(2, 2, 2), F32(2, 3), F32(2, 3)]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_div.List)
def conveter_aten__foreach_div_List(
    self: List[Tensor], other: List[Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_div.List(Tensor[] self, Tensor[] other) -> Tensor[]"""
    return ge.ForeachDivList(self, other)


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], [1., 1., 1.]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_div.ScalarList)
def conveter_aten__foreach_div_ScalarList(
    self: List[Tensor], scalars: Union[List[Number], Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_foreach_div.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]"""
    return ge.ForeachDivScalarList(self, scalars)


@register_fx_node_ge_converter(torch.ops.aten._foreach_div.Scalar_out)
def conveter_aten__foreach_div_Scalar_out(
    self: List[Tensor],
    scalar: Union[Number, Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_div.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_div.Scalar_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_div.List_out)
def conveter_aten__foreach_div_List_out(
    self: List[Tensor],
    other: List[Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_div.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_div.List_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_div.ScalarList_out)
def conveter_aten__foreach_div_ScalarList_out(
    self: List[Tensor],
    scalars: Union[List[Number], Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_div.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_div.ScalarList_out ge_converter is not supported!")
