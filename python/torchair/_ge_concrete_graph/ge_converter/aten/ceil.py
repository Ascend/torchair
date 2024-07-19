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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported([
    Support(F32(3, 4)),
    Support(F16(3, 4)),
])
@register_fx_node_ge_converter(torch.ops.aten.ceil.default)
def conveter_aten_ceil_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::ceil(Tensor self) -> Tensor"""
    return ge.Ceil(self)


@register_fx_node_ge_converter(torch.ops.aten.ceil.out)
def conveter_aten_ceil_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.ceil.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.ceil.int)
def conveter_aten_ceil_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::ceil.int(int a) -> int"""
    raise RuntimeError("torch.ops.aten.ceil.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.ceil.float)
def conveter_aten_ceil_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::ceil.float(float a) -> int"""
    raise RuntimeError("torch.ops.aten.ceil.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.ceil.Scalar)
def conveter_aten_ceil_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::ceil.Scalar(Scalar a) -> Scalar"""
    raise RuntimeError("torch.ops.aten.ceil.Scalar ge_converter is not supported!")
