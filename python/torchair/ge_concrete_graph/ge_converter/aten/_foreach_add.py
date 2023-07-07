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


@register_fx_node_ge_converter(torch.ops.aten._foreach_add.Scalar)
def conveter_aten__foreach_add_Scalar(
        self: List[Tensor],
        scalar: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_add.Scalar(Tensor[] self, Scalar scalar) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_add.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_add.List)
def conveter_aten__foreach_add_List(
        self: List[Tensor],
        other: List[Tensor],
        *,
        alpha: Union[Number, Tensor] = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_add.List(Tensor[] self, Tensor[] other, *, Scalar alpha=1) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_add.List ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_add.ScalarList)
def conveter_aten__foreach_add_ScalarList(
        self: List[Tensor],
        scalars: Union[List[Number], Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_add.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_add.ScalarList ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_add.Scalar_out)
def conveter_aten__foreach_add_Scalar_out(
        self: List[Tensor],
        scalar: Union[Number, Tensor],
        *,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_add.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_add.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_add.List_out)
def conveter_aten__foreach_add_List_out(
        self: List[Tensor],
        other: List[Tensor],
        *,
        alpha: Union[Number, Tensor] = 1,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_add.List_out(Tensor[] self, Tensor[] other, *, Scalar alpha=1, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_add.List_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_add.ScalarList_out)
def conveter_aten__foreach_add_ScalarList_out(
        self: List[Tensor],
        scalars: Union[List[Number], Tensor],
        *,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_add.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_add.ScalarList_out ge converter is not implement!")


