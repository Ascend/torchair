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


@register_fx_node_ge_converter(torch.ops.aten.log10.default)
def conveter_aten_log10_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::log10(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.log10.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log10.out)
def conveter_aten_log10_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.log10.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log10.int)
def conveter_aten_log10_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::log10.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.log10.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log10.float)
def conveter_aten_log10_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::log10.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.log10.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log10.complex)
def conveter_aten_log10_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::log10.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.log10.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log10.Scalar)
def conveter_aten_log10_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::log10.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.log10.Scalar ge_converter is not implemented!")
