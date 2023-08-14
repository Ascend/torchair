from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.floor.default)
def conveter_aten_floor_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::floor(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.floor.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.floor.out)
def conveter_aten_floor_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.floor.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.floor.int)
def conveter_aten_floor_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::floor.int(int a) -> int"""
    raise NotImplementedError("torch.ops.aten.floor.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.floor.float)
def conveter_aten_floor_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::floor.float(float a) -> int"""
    raise NotImplementedError("torch.ops.aten.floor.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.floor.Scalar)
def conveter_aten_floor_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::floor.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.floor.Scalar ge_converter is not implemented!")
