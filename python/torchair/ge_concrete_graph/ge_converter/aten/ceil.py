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
from torch import Generator, contiguous_format, inf, memory_format, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.ceil.default)
def conveter_aten_ceil_default(self: Tensor, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """NB: aten::ceil(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.ceil.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ceil.out)
def conveter_aten_ceil_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.ceil.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ceil.int)
def conveter_aten_ceil_int(a: int, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """NB: aten::ceil.int(int a) -> int"""
    raise NotImplementedError("torch.ops.aten.ceil.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ceil.float)
def conveter_aten_ceil_float(a: float, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """NB: aten::ceil.float(float a) -> int"""
    raise NotImplementedError("torch.ops.aten.ceil.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ceil.Scalar)
def conveter_aten_ceil_Scalar(a: Union[Number, Tensor], meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """NB: aten::ceil.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.ceil.Scalar ge_converter is not implemented!")
