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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcdiv_.Scalar)
def conveter_aten__foreach_addcdiv__Scalar(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    value: Union[Number, Tensor] = 1
):
    """NB: aten::_foreach_addcdiv_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_addcdiv_.Scalar ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcdiv_.ScalarList)
def conveter_aten__foreach_addcdiv__ScalarList(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    scalars: Union[List[Number], Tensor]
):
    """NB: aten::_foreach_addcdiv_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_addcdiv_.ScalarList ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcdiv_.Tensor)
def conveter_aten__foreach_addcdiv__Tensor(
    self: List[Tensor],
    tensor1: List[Tensor],
    tensor2: List[Tensor],
    scalars: Tensor
):
    """NB: aten::_foreach_addcdiv_.Tensor(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> ()"""
    raise NotImplementedError("torch.ops.aten._foreach_addcdiv_.Tensor ge_converter is not implemented!")
