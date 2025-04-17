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
        Support([I32(2, 2, 2), I32(2, 3)], [I32(2, 2, 2), I32(2, 3)], [I32(2, 2, 2), I32(2, 3)], 1),
        Support([F32(2, 2, 2), F32(2, 3)], [F32(2, 2, 2), F32(2, 3)], [F32(2, 2, 2), F32(2, 3)], 1.),
        Support([F16(2, 2, 2), F16(2, 3)], [F16(2, 2, 2), F16(2, 3)], [F16(2, 2, 2), F16(2, 3)], 1.),
        Support([BF16(2, 2, 2), BF16(2, 3)], [BF16(2, 2, 2), BF16(2, 3)], [BF16(2, 2, 2), BF16(2, 3)], 1.),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.Scalar)
def conveter_aten__foreach_addcmul_Scalar(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    value: Union[Number, Tensor] = 1,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::_foreach_addcmul.Scalar(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]"""
    if len(self) > 0:
        if self[0].dtype == DataType.DT_BF16:
            value = dtype_promote(value, target_dtype=DataType.DT_FLOAT)
        else:
            value = dtype_promote(value, target_dtype=self[0].dtype)
    return ge.ForeachAddcmulScalar(self, tensor1, tensor2, value)


@declare_supported(
    [
        Support([I32(2, 2, 2), I32(2, 3)], [I32(2, 2, 2), I32(2, 3)], [I32(2, 2, 2), I32(2, 3)], [1, 1]),
        Support([F32(2, 2, 2), F32(2, 3)], [F32(2, 2, 2), F32(2, 3)], [F32(2, 2, 2), F32(2, 3)], [1., 1.]),
        Support([F16(2, 2, 2), F16(2, 3)], [F16(2, 2, 2), F16(2, 3)], [F16(2, 2, 2), F16(2, 3)], [1., 1.]),
        Support([BF16(2, 2, 2), BF16(2, 3)], [BF16(2, 2, 2), BF16(2, 3)], [BF16(2, 2, 2), BF16(2, 3)], [1., 1.]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.ScalarList)
def conveter_aten__foreach_addcmul_ScalarList(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    scalars: Union[List[Number], Tensor],
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::_foreach_addcmul.ScalarList(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]"""
    return ge.ForeachAddcmulScalarList(self, tensor1, tensor2, scalars)


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.Tensor)
def conveter_aten__foreach_addcmul_Tensor(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    scalars: Tensor,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::_foreach_addcmul.Tensor(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> Tensor[]"""
    return ge.ForeachAddcmulList(self, tensor1, tensor2, scalars)


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.Scalar_out)
def conveter_aten__foreach_addcmul_Scalar_out(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    value: Union[Number, Tensor] = 1,
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_addcmul.Scalar_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_addcmul.Scalar_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.ScalarList_out)
def conveter_aten__foreach_addcmul_ScalarList_out(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    scalars: Union[List[Number], Tensor],
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_addcmul.ScalarList_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_addcmul.ScalarList_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.Tensor_out)
def conveter_aten__foreach_addcmul_Tensor_out(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    scalars: Tensor,
    *,
    out: List[Tensor] = None
):
    """NB: aten::_foreach_addcmul.Tensor_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars, *, Tensor(a!)[] out) -> ()"""
    raise RuntimeError("torch.ops.aten._foreach_addcmul.Tensor_out ge_converter is not supported!")
