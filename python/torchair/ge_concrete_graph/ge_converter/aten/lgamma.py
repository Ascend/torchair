import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
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


@register_fx_node_ge_converter(torch.ops.aten.lgamma.default)
def conveter_aten_lgamma_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lgamma(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.lgamma.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.out)
def conveter_aten_lgamma_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.lgamma.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.int)
def conveter_aten_lgamma_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lgamma.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.lgamma.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.float)
def conveter_aten_lgamma_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lgamma.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.lgamma.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.Scalar)
def conveter_aten_lgamma_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::lgamma.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.lgamma.Scalar ge converter is not implement!")


