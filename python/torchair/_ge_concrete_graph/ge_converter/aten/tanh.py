from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported([
    Support(F32(1024)),
    Support(F16(1024)),
    Support(F32(10, 40, 1024)),
    Support(F16(10, 40, 1024)),

])
@register_fx_node_ge_converter(torch.ops.aten.tanh.default)
def conveter_aten_tanh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::tanh(Tensor self) -> Tensor"""
    return ge.Tanh(self)


@register_fx_node_ge_converter(torch.ops.aten.tanh.out)
def conveter_aten_tanh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.tanh.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.tanh.int)
def conveter_aten_tanh_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::tanh.int(int a) -> float"""
    raise RuntimeError("torch.ops.aten.tanh.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.tanh.float)
def conveter_aten_tanh_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::tanh.float(float a) -> float"""
    raise RuntimeError("torch.ops.aten.tanh.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.tanh.complex)
def conveter_aten_tanh_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::tanh.complex(complex a) -> complex"""
    raise RuntimeError("torch.ops.aten.tanh.complex ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.tanh.Scalar)
def conveter_aten_tanh_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::tanh.Scalar(Scalar a) -> Scalar"""
    raise RuntimeError("torch.ops.aten.tanh.Scalar ge_converter is not supported!")
