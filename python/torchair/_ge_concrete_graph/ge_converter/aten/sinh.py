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
    Support(F32(3, 4, 5)),
    Support(F16(3, 4, 5)),
])
@register_fx_node_ge_converter(torch.ops.aten.sinh.default)
def conveter_aten_sinh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sinh(Tensor self) -> Tensor"""
    return ge.Sinh(self)


@register_fx_node_ge_converter(torch.ops.aten.sinh.out)
def conveter_aten_sinh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.sinh.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sinh.int)
def conveter_aten_sinh_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::sinh.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.sinh.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sinh.float)
def conveter_aten_sinh_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::sinh.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.sinh.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sinh.complex)
def conveter_aten_sinh_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::sinh.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.sinh.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sinh.Scalar)
def conveter_aten_sinh_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::sinh.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.sinh.Scalar ge_converter is not implemented!")
