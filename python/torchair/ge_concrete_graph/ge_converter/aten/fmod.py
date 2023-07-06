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


@register_fx_node_ge_converter(torch.ops.aten.fmod.Tensor)
def conveter_aten_fmod_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.fmod.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fmod.Scalar)
def conveter_aten_fmod_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.fmod.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fmod.Tensor_out)
def conveter_aten_fmod_Tensor_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.fmod.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fmod.Scalar_out)
def conveter_aten_fmod_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.fmod.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fmod.int)
def conveter_aten_fmod_int(
        a: int,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::fmod.int(int a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.fmod.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fmod.float)
def conveter_aten_fmod_float(
        a: float,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::fmod.float(float a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.fmod.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fmod.int_float)
def conveter_aten_fmod_int_float(
        a: int,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::fmod.int_float(int a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.fmod.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fmod.float_int)
def conveter_aten_fmod_float_int(
        a: float,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::fmod.float_int(float a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.fmod.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.fmod.default)
def conveter_aten_fmod_default(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::fmod(Scalar a, Scalar b) -> float """
    raise NotImplementedError("torch.ops.aten.fmod.default ge converter is not implement!")


