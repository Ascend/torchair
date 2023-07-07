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


@register_fx_node_ge_converter(torch.ops.aten.xlogy.Tensor)
def conveter_aten_xlogy_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.xlogy.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy.Scalar_Other)
def conveter_aten_xlogy_Scalar_Other(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.xlogy.Scalar_Other ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy.Scalar_Self)
def conveter_aten_xlogy_Scalar_Self(
        self: Union[Number, Tensor],
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::xlogy.Scalar_Self(Scalar self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.xlogy.Scalar_Self ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy.OutTensor)
def conveter_aten_xlogy_OutTensor(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::xlogy.OutTensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.xlogy.OutTensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy.OutScalar_Self)
def conveter_aten_xlogy_OutScalar_Self(
        self: Union[Number, Tensor],
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::xlogy.OutScalar_Self(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.xlogy.OutScalar_Self ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.xlogy.OutScalar_Other)
def conveter_aten_xlogy_OutScalar_Other(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::xlogy.OutScalar_Other(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.xlogy.OutScalar_Other ge converter is not implement!")


