import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcdiv_.Scalar)
def conveter_aten__foreach_addcdiv__Scalar(
        self: List[Tensor],
        tensor1: List[Tensor],
        tensor2: List[Tensor],
        value: Union[Number, Tensor] = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_addcdiv_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_addcdiv_.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcdiv_.ScalarList)
def conveter_aten__foreach_addcdiv__ScalarList(
        self: List[Tensor],
        tensor1: List[Tensor],
        tensor2: List[Tensor],
        scalars: Union[List[Number], Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_addcdiv_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_addcdiv_.ScalarList ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcdiv_.Tensor)
def conveter_aten__foreach_addcdiv__Tensor(
        self: List[Tensor],
        tensor1: List[Tensor],
        tensor2: List[Tensor],
        scalars: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_addcdiv_.Tensor(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_addcdiv_.Tensor ge converter is not implement!")


