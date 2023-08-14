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


@register_fx_node_ge_converter(torch.ops.aten.asin.default)
def conveter_aten_asin_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::asin(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.asin.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asin.out)
def conveter_aten_asin_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.asin.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asin.int)
def conveter_aten_asin_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::asin.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.asin.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asin.float)
def conveter_aten_asin_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::asin.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.asin.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asin.complex)
def conveter_aten_asin_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::asin.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.asin.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asin.Scalar)
def conveter_aten_asin_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::asin.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.asin.Scalar ge_converter is not implemented!")
