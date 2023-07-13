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


@register_fx_node_ge_converter(torch.ops.aten.lt.Tensor)
def conveter_aten_lt_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lt.Tensor(Tensor self, Tensor other) -> Tensor """
    return ge.Less(self, other)


@register_fx_node_ge_converter(torch.ops.aten.lt.Scalar)
def conveter_aten_lt_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lt.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.lt.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lt.Scalar_out)
def conveter_aten_lt_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.lt.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lt.Tensor_out)
def conveter_aten_lt_Tensor_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.lt.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lt.int)
def conveter_aten_lt_int(
        a: int,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lt.int(int a, int b) -> bool """
    raise NotImplementedError("torch.ops.aten.lt.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lt.float)
def conveter_aten_lt_float(
        a: float,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lt.float(float a, float b) -> bool """
    raise NotImplementedError("torch.ops.aten.lt.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lt.int_float)
def conveter_aten_lt_int_float(
        a: int,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lt.int_float(int a, float b) -> bool """
    raise NotImplementedError("torch.ops.aten.lt.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lt.float_int)
def conveter_aten_lt_float_int(
        a: float,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lt.float_int(float a, int b) -> bool """
    raise NotImplementedError("torch.ops.aten.lt.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lt.default)
def conveter_aten_lt_default(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lt(Scalar a, Scalar b) -> bool """
    raise NotImplementedError("torch.ops.aten.lt.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lt.str)
def conveter_aten_lt_str(
        a: str,
        b: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lt.str(str a, str b) -> bool """
    raise NotImplementedError("torch.ops.aten.lt.str ge converter is not implement!")


