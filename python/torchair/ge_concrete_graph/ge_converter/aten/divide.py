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


@register_fx_node_ge_converter(torch.ops.aten.divide.Tensor)
def conveter_aten_divide_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::divide.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.divide.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.divide.Scalar)
def conveter_aten_divide_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::divide.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.divide.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.divide.Tensor_mode)
def conveter_aten_divide_Tensor_mode(
        self: Tensor,
        other: Tensor,
        *,
        rounding_mode: Optional[str],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor """
    raise NotImplementedError("torch.ops.aten.divide.Tensor_mode ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.divide.Scalar_mode)
def conveter_aten_divide_Scalar_mode(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        rounding_mode: Optional[str],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor """
    raise NotImplementedError("torch.ops.aten.divide.Scalar_mode ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.divide.out)
def conveter_aten_divide_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.divide.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.divide.out_mode)
def conveter_aten_divide_out_mode(
        self: Tensor,
        other: Tensor,
        *,
        rounding_mode: Optional[str],
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.divide.out_mode ge converter is not implement!")


