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
@register_fx_node_ge_converter(torch.ops.aten.tan.default)
def conveter_aten_tan_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::tan(Tensor self) -> Tensor"""
    return ge.Tan(self)


@register_fx_node_ge_converter(torch.ops.aten.tan.out)
def conveter_aten_tan_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.tan.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.tan.int)
def conveter_aten_tan_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::tan.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.tan.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.tan.float)
def conveter_aten_tan_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::tan.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.tan.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.tan.complex)
def conveter_aten_tan_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::tan.complex(complex a) -> complex"""
    raise NotImplementedError("torch.ops.aten.tan.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.tan.Scalar)
def conveter_aten_tan_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::tan.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.tan.Scalar ge_converter is not implemented!")
