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


@register_fx_node_ge_converter(torch.ops.aten.cosh.default)
def conveter_aten_cosh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::cosh(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.cosh.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cosh.out)
def conveter_aten_cosh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cosh.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cosh.int)
def conveter_aten_cosh_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::cosh.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.cosh.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cosh.float)
def conveter_aten_cosh_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::cosh.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.cosh.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cosh.complex)
def conveter_aten_cosh_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::cosh.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.cosh.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cosh.Scalar)
def conveter_aten_cosh_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::cosh.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.cosh.Scalar ge_converter is not implemented!")
