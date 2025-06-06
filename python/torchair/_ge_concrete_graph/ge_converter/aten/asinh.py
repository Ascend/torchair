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
@register_fx_node_ge_converter(torch.ops.aten.asinh.default)
def conveter_aten_asinh_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::asinh(Tensor self) -> Tensor"""
    if meta_outputs:
        self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    return ge.Asinh(self)


@register_fx_node_ge_converter(torch.ops.aten.asinh.out)
def conveter_aten_asinh_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.asinh.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asinh.int)
def conveter_aten_asinh_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::asinh.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.asinh.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asinh.float)
def conveter_aten_asinh_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::asinh.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.asinh.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asinh.complex)
def conveter_aten_asinh_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::asinh.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.asinh.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.asinh.Scalar)
def conveter_aten_asinh_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::asinh.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.asinh.Scalar ge_converter is not implemented!")
