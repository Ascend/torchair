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


@register_fx_node_ge_converter(torch.ops.aten.le.Tensor)
def conveter_aten_le_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::le.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.le.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.le.Scalar)
def conveter_aten_le_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::le.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.le.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.le.Scalar_out)
def conveter_aten_le_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.le.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.le.Tensor_out)
def conveter_aten_le_Tensor_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.le.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.le.int)
def conveter_aten_le_int(
        a: int,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::le.int(int a, int b) -> bool """
    raise NotImplementedError("torch.ops.aten.le.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.le.float)
def conveter_aten_le_float(
        a: float,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::le.float(float a, float b) -> bool """
    raise NotImplementedError("torch.ops.aten.le.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.le.int_float)
def conveter_aten_le_int_float(
        a: int,
        b: float,
        meta_outputs: Any = None):
    """ NB: aten::le.int_float(int a, float b) -> bool """
    raise NotImplementedError("torch.ops.aten.le.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.le.float_int)
def conveter_aten_le_float_int(
        a: float,
        b: int,
        meta_outputs: Any = None):
    """ NB: aten::le.float_int(float a, int b) -> bool """
    raise NotImplementedError("torch.ops.aten.le.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.le.default)
def conveter_aten_le_default(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::le(Scalar a, Scalar b) -> bool """
    raise NotImplementedError("torch.ops.aten.le.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.le.str)
def conveter_aten_le_str(
        a: str,
        b: str,
        meta_outputs: Any = None):
    """ NB: aten::le.str(str a, str b) -> bool """
    raise NotImplementedError("torch.ops.aten.le.str ge converter is not implement!")


