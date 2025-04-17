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
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(1024)),
    Support(F16(1024)),
    Support(F32(10, 40, 1024)),
    Support(F16(10, 40, 1024)),
])
@register_fx_node_ge_converter(torch.ops.aten.atanh.default)
def conveter_aten_atanh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::atanh(Tensor self) -> Tensor"""
    return ge.Atanh(self)


@register_fx_node_ge_converter(torch.ops.aten.atanh.out)
def conveter_aten_atanh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.atanh.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.int)
def conveter_aten_atanh_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::atanh.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.atanh.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.float)
def conveter_aten_atanh_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::atanh.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.atanh.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.complex)
def conveter_aten_atanh_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::atanh.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.atanh.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.Scalar)
def conveter_aten_atanh_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::atanh.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.atanh.Scalar ge_converter is not implemented!")
