import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
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


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul_.Scalar)
def conveter_aten__foreach_mul__Scalar(
        self: List[Tensor],
        scalar: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_mul_.Scalar(Tensor(a!)[] self, Scalar scalar) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_mul_.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul_.List)
def conveter_aten__foreach_mul__List(
        self: List[Tensor],
        other: List[Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_mul_.List(Tensor(a!)[] self, Tensor[] other) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_mul_.List ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul_.ScalarList)
def conveter_aten__foreach_mul__ScalarList(
        self: List[Tensor],
        scalars: Union[List[Number], Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_mul_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_mul_.ScalarList ge converter is not implement!")


