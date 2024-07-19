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
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote

@declare_supported([
    Support(F32(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.log.default)
def conveter_aten_log_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::log(Tensor self) -> Tensor"""
    return ge.Log(self)


@register_fx_node_ge_converter(torch.ops.aten.log.out)
def conveter_aten_log_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.log.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.int)
def conveter_aten_log_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::log.int(int a) -> float"""
    raise RuntimeError("torch.ops.aten.log.int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.float)
def conveter_aten_log_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::log.float(float a) -> float"""
    raise RuntimeError("torch.ops.aten.log.float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.complex)
def conveter_aten_log_complex(a: complex, meta_outputs: TensorSpec = None):
    """NB: aten::log.complex(complex a) -> complex"""
    raise RuntimeError("torch.ops.aten.log.complex ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.Scalar)
def conveter_aten_log_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::log.Scalar(Scalar a) -> Scalar"""
    raise RuntimeError("torch.ops.aten.log.Scalar ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.int_int)
def conveter_aten_log_int_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::log.int_int(int a, int b) -> float"""
    raise RuntimeError("torch.ops.aten.log.int_int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.float_float)
def conveter_aten_log_float_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::log.float_float(float a, float b) -> float"""
    raise RuntimeError("torch.ops.aten.log.float_float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.complex_complex)
def conveter_aten_log_complex_complex(a: complex, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::log.complex_complex(complex a, complex b) -> complex"""
    raise RuntimeError("torch.ops.aten.log.complex_complex ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.int_float)
def conveter_aten_log_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::log.int_float(int a, float b) -> float"""
    raise RuntimeError("torch.ops.aten.log.int_float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.float_int)
def conveter_aten_log_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::log.float_int(float a, int b) -> float"""
    raise RuntimeError("torch.ops.aten.log.float_int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.int_complex)
def conveter_aten_log_int_complex(a: int, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::log.int_complex(int a, complex b) -> complex"""
    raise RuntimeError("torch.ops.aten.log.int_complex ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.complex_int)
def conveter_aten_log_complex_int(a: complex, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::log.complex_int(complex a, int b) -> complex"""
    raise RuntimeError("torch.ops.aten.log.complex_int ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.float_complex)
def conveter_aten_log_float_complex(a: float, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::log.float_complex(float a, complex b) -> complex"""
    raise RuntimeError("torch.ops.aten.log.float_complex ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.complex_float)
def conveter_aten_log_complex_float(a: complex, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::log.complex_float(complex a, float b) -> complex"""
    raise RuntimeError("torch.ops.aten.log.complex_float ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.log.Scalar_Scalar)
def conveter_aten_log_Scalar_Scalar(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::log.Scalar_Scalar(Scalar a, Scalar b) -> float"""
    raise RuntimeError("torch.ops.aten.log.Scalar_Scalar ge_converter is not supported!")
