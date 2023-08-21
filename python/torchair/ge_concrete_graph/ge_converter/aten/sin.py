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


@register_fx_node_ge_converter(torch.ops.aten.sin.default)
def conveter_aten_sin_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sin(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.sin.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sin.out)
def conveter_aten_sin_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sin.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sin.int)
def conveter_aten_sin_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::sin.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.sin.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sin.float)
def conveter_aten_sin_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::sin.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.sin.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sin.complex)
def conveter_aten_sin_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::sin.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.sin.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sin.Scalar)
def conveter_aten_sin_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::sin.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.sin.Scalar ge_converter is not implemented!")
