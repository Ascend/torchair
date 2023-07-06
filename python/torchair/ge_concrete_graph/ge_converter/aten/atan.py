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


@register_fx_node_ge_converter(torch.ops.aten.atan.default)
def conveter_aten_atan_default(
        self: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::atan(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.atan.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan.out)
def conveter_aten_atan_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.atan.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan.int)
def conveter_aten_atan_int(
        a: int,
        meta_outputs: Any = None):
    """ NB: aten::atan.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.atan.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan.float)
def conveter_aten_atan_float(
        a: float,
        meta_outputs: Any = None):
    """ NB: aten::atan.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.atan.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan.complex)
def conveter_aten_atan_complex(
        a: complex,
        meta_outputs: Any = None):
    """ NB: aten::atan.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.atan.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atan.Scalar)
def conveter_aten_atan_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::atan.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.atan.Scalar ge converter is not implement!")


