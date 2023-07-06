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


@register_fx_node_ge_converter(torch.ops.aten.acos.default)
def conveter_aten_acos_default(
        self: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::acos(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.acos.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.acos.out)
def conveter_aten_acos_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.acos.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.acos.int)
def conveter_aten_acos_int(
        a: int,
        meta_outputs: Any = None):
    """ NB: aten::acos.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.acos.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.acos.float)
def conveter_aten_acos_float(
        a: float,
        meta_outputs: Any = None):
    """ NB: aten::acos.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.acos.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.acos.complex)
def conveter_aten_acos_complex(
        a: complex,
        meta_outputs: Any = None):
    """ NB: aten::acos.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.acos.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.acos.Scalar)
def conveter_aten_acos_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::acos.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.acos.Scalar ge converter is not implement!")


