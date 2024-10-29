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
@register_fx_node_ge_converter(torch.ops.aten._foreach_minimum.Scalar)
def conveter_aten__foreach_minimum_scalar(
    self: List[Tensor],
    scalar: Union[Number, Tensor],
    meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_lerp.Scalar(Tensor[] self, Tensor[] tensors1, Scalar weight) -> Tensor[]"""
    return ge.ForeachMinimumScalar(self, scalar)


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], [1., 1., 1.]),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], [1., 1., 1.]),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], [1., 1., 1.]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_minimum.ScalarList)
def conveter_aten__foreach_minimum_scalarlist(
    self: List[Tensor],
    scalars: Union[List[Number], Tensor],
    meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_lerp.Scalar(Tensor[] self, Tensor[] tensors1, Scalar weight) -> Tensor[]"""
    return ge.ForeachMinimumScalarList(self, scalars)


@declare_supported(
    [
        Support([F32(2, 2, 2), F32(2, 3), F32(2, 3)], [F32(2, 2, 2), F32(2, 3), F32(2, 3)]),
        Support([F16(2, 2, 2), F16(2, 3), F16(2, 3)], [F16(2, 2, 2), F16(2, 3), F16(2, 3)]),
        Support([BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)], [BF16(2, 2, 2), BF16(2, 3), BF16(2, 3)]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_minimum.List)
def conveter_aten__foreach_minimum_list(
    self: List[Tensor],
    other: List[Tensor],
    meta_outputs: List[TensorSpec] = None):
    """NB: aten::_foreach_lerp.Scalar(Tensor[] self, Tensor[] tensors1, Scalar weight) -> Tensor[]"""
    return ge.ForeachMinimumList(self, other)
