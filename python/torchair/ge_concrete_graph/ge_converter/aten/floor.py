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


@register_fx_node_ge_converter(torch.ops.aten.floor.default)
def conveter_aten_floor_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::floor(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.floor.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.floor.out)
def conveter_aten_floor_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.floor.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.floor.int)
def conveter_aten_floor_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::floor.int(int a) -> int """
    raise NotImplementedError("torch.ops.aten.floor.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.floor.float)
def conveter_aten_floor_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::floor.float(float a) -> int """
    raise NotImplementedError("torch.ops.aten.floor.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.floor.Scalar)
def conveter_aten_floor_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::floor.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.floor.Scalar ge converter is not implement!")


