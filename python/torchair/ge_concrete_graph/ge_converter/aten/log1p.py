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


@register_fx_node_ge_converter(torch.ops.aten.log1p.default)
def conveter_aten_log1p_default(
        self: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::log1p(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.log1p.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log1p.out)
def conveter_aten_log1p_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.log1p.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log1p.int)
def conveter_aten_log1p_int(
        a: int,
        meta_outputs: Any = None):
    """ NB: aten::log1p.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.log1p.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log1p.float)
def conveter_aten_log1p_float(
        a: float,
        meta_outputs: Any = None):
    """ NB: aten::log1p.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.log1p.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log1p.Scalar)
def conveter_aten_log1p_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Any = None):
    """ NB: aten::log1p.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.log1p.Scalar ge converter is not implement!")


