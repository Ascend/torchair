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


@register_fx_node_ge_converter(torch.ops.aten.ge.Tensor)
def conveter_aten_ge_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::ge.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.ge.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.ge.Scalar)
def conveter_aten_ge_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::ge.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.ge.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.ge.Scalar_out)
def conveter_aten_ge_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.ge.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.ge.Tensor_out)
def conveter_aten_ge_Tensor_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.ge.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.ge.int)
def conveter_aten_ge_int(
        a: int,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::ge.int(int a, int b) -> bool """
    raise NotImplementedError("torch.ops.aten.ge.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.ge.float)
def conveter_aten_ge_float(
        a: float,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::ge.float(float a, float b) -> bool """
    raise NotImplementedError("torch.ops.aten.ge.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.ge.int_float)
def conveter_aten_ge_int_float(
        a: int,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::ge.int_float(int a, float b) -> bool """
    raise NotImplementedError("torch.ops.aten.ge.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.ge.float_int)
def conveter_aten_ge_float_int(
        a: float,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::ge.float_int(float a, int b) -> bool """
    raise NotImplementedError("torch.ops.aten.ge.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.ge.default)
def conveter_aten_ge_default(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::ge(Scalar a, Scalar b) -> bool """
    raise NotImplementedError("torch.ops.aten.ge.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.ge.str)
def conveter_aten_ge_str(
        a: str,
        b: str,
        meta_outputs: Any = None):
    """ NB: aten::ge.str(str a, str b) -> bool """
    raise NotImplementedError("torch.ops.aten.ge.str ge converter is not implement!")


