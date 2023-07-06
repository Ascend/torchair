import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
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


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul.Scalar)
def conveter_aten__foreach_mul_Scalar(
        self: List[Tensor],
        scalar: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::_foreach_mul.Scalar(Tensor[] self, Scalar scalar) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_mul.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul.List)
def conveter_aten__foreach_mul_List(
        self: List[Tensor],
        other: List[Tensor],
        meta_outputs: Any = None):
    """ NB: aten::_foreach_mul.List(Tensor[] self, Tensor[] other) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_mul.List ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul.ScalarList)
def conveter_aten__foreach_mul_ScalarList(
        self: List[Tensor],
        scalars: Union[List[Number], Tensor],
        meta_outputs: Any = None):
    """ NB: aten::_foreach_mul.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_mul.ScalarList ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul.Scalar_out)
def conveter_aten__foreach_mul_Scalar_out(
        self: List[Tensor],
        scalar: Union[Number, Tensor],
        *,
        out: List[Tensor] = None,
        meta_outputs: Any = None):
    """ NB: aten::_foreach_mul.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_mul.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul.List_out)
def conveter_aten__foreach_mul_List_out(
        self: List[Tensor],
        other: List[Tensor],
        *,
        out: List[Tensor] = None,
        meta_outputs: Any = None):
    """ NB: aten::_foreach_mul.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_mul.List_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_mul.ScalarList_out)
def conveter_aten__foreach_mul_ScalarList_out(
        self: List[Tensor],
        scalars: Union[List[Number], Tensor],
        *,
        out: List[Tensor] = None,
        meta_outputs: Any = None):
    """ NB: aten::_foreach_mul.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_mul.ScalarList_out ge converter is not implement!")


