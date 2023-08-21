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


@register_fx_node_ge_converter(torch.ops.aten.atan.default)
def conveter_aten_atan_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::atan(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.atan.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan.out)
def conveter_aten_atan_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.atan.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan.int)
def conveter_aten_atan_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::atan.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.atan.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan.float)
def conveter_aten_atan_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::atan.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.atan.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan.complex)
def conveter_aten_atan_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::atan.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.atan.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atan.Scalar)
def conveter_aten_atan_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::atan.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.atan.Scalar ge_converter is not implemented!")
