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


@register_fx_node_ge_converter(torch.ops.aten.tanh.default)
def conveter_aten_tanh_default(
        self: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::tanh(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.tanh.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.tanh.out)
def conveter_aten_tanh_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.tanh.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.tanh.int)
def conveter_aten_tanh_int(
        a: int,
        meta_outputs: Any = None):
    """ NB: aten::tanh.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.tanh.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.tanh.float)
def conveter_aten_tanh_float(
        a: float,
        meta_outputs: Any = None):
    """ NB: aten::tanh.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.tanh.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.tanh.complex)
def conveter_aten_tanh_complex(
        a: complex,
        meta_outputs: Any = None):
    """ NB: aten::tanh.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.tanh.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.tanh.Scalar)
def conveter_aten_tanh_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::tanh.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.tanh.Scalar ge converter is not implement!")


