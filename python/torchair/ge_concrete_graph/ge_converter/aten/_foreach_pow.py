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


@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.List)
def conveter_aten__foreach_pow_List(
        self: List[Tensor],
        exponent: List[Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_pow.List(Tensor[] self, Tensor[] exponent) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_pow.List ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.Scalar)
def conveter_aten__foreach_pow_Scalar(
        self: List[Tensor],
        exponent: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_pow.Scalar(Tensor[] self, Scalar exponent) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_pow.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.ScalarList)
def conveter_aten__foreach_pow_ScalarList(
        self: List[Tensor],
        exponent: Union[List[Number], Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_pow.ScalarList(Tensor[] self, Scalar[] exponent) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_pow.ScalarList ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.ScalarAndTensor)
def conveter_aten__foreach_pow_ScalarAndTensor(
        self: Union[Number, Tensor],
        exponent: List[Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_pow.ScalarAndTensor(Scalar self, Tensor[] exponent) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_pow.ScalarAndTensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.List_out)
def conveter_aten__foreach_pow_List_out(
        self: List[Tensor],
        exponent: List[Tensor],
        *,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_pow.List_out(Tensor[] self, Tensor[] exponent, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_pow.List_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.Scalar_out)
def conveter_aten__foreach_pow_Scalar_out(
        self: List[Tensor],
        exponent: Union[Number, Tensor],
        *,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_pow.Scalar_out(Tensor[] self, Scalar exponent, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_pow.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_pow.ScalarList_out)
def conveter_aten__foreach_pow_ScalarList_out(
        self: List[Tensor],
        exponent: Union[List[Number], Tensor],
        *,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_pow.ScalarList_out(Tensor[] self, Scalar[] exponent, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_pow.ScalarList_out ge converter is not implement!")


