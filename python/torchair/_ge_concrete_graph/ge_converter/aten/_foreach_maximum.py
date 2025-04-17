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
        Support([I32(2, 2, 2), I32(2, 3), I32(2, 3)], 1),
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], 1.),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], 1.),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], 1.),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_maximum.Scalar)
def conveter_aten__foreach_maximum_scalar(
    self: List[Tensor],
    scalar: Union[Number, Tensor],
    meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_maximum.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]"""
    if len(self) > 0:
        if self[0].dtype == DataType.DT_BF16:
            scalar = dtype_promote(scalar, target_dtype=DataType.DT_FLOAT)
        else:
            scalar = dtype_promote(scalar, target_dtype=self[0].dtype)
    return ge.ForeachMaximumScalar(self, scalar)


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], [1., 1., 1.]),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], [1., 1., 1.]),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], [1., 1., 1.]),
        Support([I32(2, 2, 2), I32(2, 3)], [1, 1]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_maximum.ScalarList)
def conveter_aten__foreach_maximum_scalarlist(
    self: List[Tensor],
    scalars: Union[List[Number], Tensor],
    meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_maximum.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]"""
    if len(scalars) > 0 and isinstance(scalars[0], int):
        scalars = dtype_promote(scalars, target_dtype=DataType.DT_INT64)
    return ge.ForeachMaximumScalarList(self, scalars)


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], [F32(2, 2, 2), F32(2, 3), F32(2, 3)]),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], [F16(2, 2, 2), F16(2, 3), F16(2, 3)]),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], [BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_maximum.List)
def conveter_aten__foreach_maximum_list(
    self: List[Tensor],
    other: List[Tensor],
    meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_maximum.List(Tensor[] self, Tensor[] other) -> Tensor[]"""
    return ge.ForeachMaximumList(self, other)
