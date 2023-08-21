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


@register_fx_node_ge_converter(torch.ops.aten.cos.default)
def conveter_aten_cos_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::cos(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.cos.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cos.out)
def conveter_aten_cos_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cos.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cos.int)
def conveter_aten_cos_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::cos.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.cos.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cos.float)
def conveter_aten_cos_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::cos.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.cos.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cos.complex)
def conveter_aten_cos_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::cos.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.cos.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cos.Scalar)
def conveter_aten_cos_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::cos.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.cos.Scalar ge_converter is not implemented!")
