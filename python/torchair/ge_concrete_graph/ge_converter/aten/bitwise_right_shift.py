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


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Tensor)
def conveter_aten_bitwise_right_shift_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Tensor_Scalar)
def conveter_aten_bitwise_right_shift_Tensor_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Tensor_Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Scalar_Tensor)
def conveter_aten_bitwise_right_shift_Scalar_Tensor(
        self: Union[Number, Tensor],
        other: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Scalar_Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Tensor_out)
def conveter_aten_bitwise_right_shift_Tensor_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Tensor_Scalar_out)
def conveter_aten_bitwise_right_shift_Tensor_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::bitwise_right_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Tensor_Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.bitwise_right_shift.Scalar_Tensor_out)
def conveter_aten_bitwise_right_shift_Scalar_Tensor_out(
        self: Union[Number, Tensor],
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::bitwise_right_shift.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.bitwise_right_shift.Scalar_Tensor_out ge converter is not implement!")


