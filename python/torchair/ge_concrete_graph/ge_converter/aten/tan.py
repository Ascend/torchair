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


@register_fx_node_ge_converter(torch.ops.aten.tan.default)
def conveter_aten_tan_default(
        self: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::tan(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.tan.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.tan.out)
def conveter_aten_tan_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.tan.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.tan.int)
def conveter_aten_tan_int(
        a: int,
        meta_outputs: Any = None):
    """ NB: aten::tan.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.tan.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.tan.float)
def conveter_aten_tan_float(
        a: float,
        meta_outputs: Any = None):
    """ NB: aten::tan.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.tan.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.tan.complex)
def conveter_aten_tan_complex(
        a: complex,
        meta_outputs: Any = None):
    """ NB: aten::tan.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.tan.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.tan.Scalar)
def conveter_aten_tan_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::tan.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.tan.Scalar ge converter is not implement!")


