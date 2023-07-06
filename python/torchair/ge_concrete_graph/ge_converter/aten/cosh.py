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


@register_fx_node_ge_converter(torch.ops.aten.cosh.default)
def conveter_aten_cosh_default(
        self: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::cosh(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.cosh.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cosh.out)
def conveter_aten_cosh_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.cosh.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cosh.int)
def conveter_aten_cosh_int(
        a: int,
        meta_outputs: Any = None):
    """ NB: aten::cosh.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.cosh.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cosh.float)
def conveter_aten_cosh_float(
        a: float,
        meta_outputs: Any = None):
    """ NB: aten::cosh.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.cosh.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cosh.complex)
def conveter_aten_cosh_complex(
        a: complex,
        meta_outputs: Any = None):
    """ NB: aten::cosh.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.cosh.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cosh.Scalar)
def conveter_aten_cosh_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::cosh.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.cosh.Scalar ge converter is not implement!")


