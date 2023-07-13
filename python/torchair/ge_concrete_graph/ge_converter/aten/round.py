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


@register_fx_node_ge_converter(torch.ops.aten.round.default)
def conveter_aten_round_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::round(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.round.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.round.decimals)
def conveter_aten_round_decimals(
        self: Tensor,
        *,
        decimals: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::round.decimals(Tensor self, *, int decimals) -> Tensor """
    raise NotImplementedError("torch.ops.aten.round.decimals ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.round.out)
def conveter_aten_round_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.round.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.round.decimals_out)
def conveter_aten_round_decimals_out(
        self: Tensor,
        *,
        decimals: int,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::round.decimals_out(Tensor self, *, int decimals, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.round.decimals_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.round.int)
def conveter_aten_round_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::round.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.round.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.round.float)
def conveter_aten_round_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::round.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.round.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.round.Scalar)
def conveter_aten_round_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::round.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.round.Scalar ge converter is not implement!")


