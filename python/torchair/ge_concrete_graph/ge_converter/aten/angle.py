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


@register_fx_node_ge_converter(torch.ops.aten.angle.default)
def conveter_aten_angle_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::angle(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.angle.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.angle.out)
def conveter_aten_angle_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.angle.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.angle.int)
def conveter_aten_angle_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::angle.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.angle.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.angle.float)
def conveter_aten_angle_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::angle.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.angle.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.angle.complex)
def conveter_aten_angle_complex(
        a: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::angle.complex(complex a) -> float """
    raise NotImplementedError("torch.ops.aten.angle.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.angle.Scalar)
def conveter_aten_angle_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::angle.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.angle.Scalar ge converter is not implement!")


