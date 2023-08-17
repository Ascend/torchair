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
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote

@declare_supported([
    Support(F32(2,2), F32(2,2)),
])
@register_fx_node_ge_converter(torch.ops.aten.ne.Tensor)
def conveter_aten_ne_Tensor(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::ne.Tensor(Tensor self, Tensor other) -> Tensor"""
    if self.dtype != other.dtype:
        value = ge.Cast(ge.Const(0.), dst_type=torch_type_to_ge_type(torch.bool))
        return ge.Fill(ge.Shape(self), value)
    return ge.NotEqual(self, other)


@declare_supported([
    Support(F32(12, 384), 1),
])
@register_fx_node_ge_converter(torch.ops.aten.ne.Scalar)
def conveter_aten_ne_Scalar(
    self: Tensor, other: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::ne.Scalar(Tensor self, Scalar other) -> Tensor"""
    return ge.NotEqual(self, other)


@register_fx_node_ge_converter(torch.ops.aten.ne.Scalar_out)
def conveter_aten_ne_Scalar_out(
    self: Tensor,
    other: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.ne.Scalar_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.Tensor_out)
def conveter_aten_ne_Tensor_out(
    self: Tensor, other: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.ne.Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.int_list)
def conveter_aten_ne_int_list(a: List[int], b: List[int], meta_outputs: TensorSpec = None):
    """NB: aten::ne.int_list(int[] a, int[] b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.int_list ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.device)
def conveter_aten_ne_device(a: Device, b: Device, meta_outputs: TensorSpec = None):
    """NB: aten::ne.device(Device a, Device b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.device ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.bool)
def conveter_aten_ne_bool(a: bool, b: bool, meta_outputs: TensorSpec = None):
    """NB: aten::ne.bool(bool a, bool b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.bool ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.int)
def conveter_aten_ne_int(a: int, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::ne.int(int a, int b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.complex)
def conveter_aten_ne_complex(a: complex, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::ne.complex(complex a, complex b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.float)
def conveter_aten_ne_float(a: float, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::ne.float(float a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.int_float)
def conveter_aten_ne_int_float(a: int, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::ne.int_float(int a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.int_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.float_int)
def conveter_aten_ne_float_int(a: float, b: int, meta_outputs: TensorSpec = None):
    """NB: aten::ne.float_int(float a, int b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.float_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.float_complex)
def conveter_aten_ne_float_complex(a: float, b: complex, meta_outputs: TensorSpec = None):
    """NB: aten::ne.float_complex(float a, complex b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.float_complex ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.complex_float)
def conveter_aten_ne_complex_float(a: complex, b: float, meta_outputs: TensorSpec = None):
    """NB: aten::ne.complex_float(complex a, float b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.complex_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.default)
def conveter_aten_ne_default(
    a: Union[Number, Tensor], b: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::ne(Scalar a, Scalar b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.str)
def conveter_aten_ne_str(a: str, b: str, meta_outputs: TensorSpec = None):
    """NB: aten::ne.str(str a, str b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.str ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.float_list)
def conveter_aten_ne_float_list(
    a: List[float], b: List[float], meta_outputs: TensorSpec = None
):
    """NB: aten::ne.float_list(float[] a, float[] b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.float_list ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.Tensor_list)
def conveter_aten_ne_Tensor_list(
    a: List[Tensor], b: List[Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::ne.Tensor_list(Tensor[] a, Tensor[] b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.Tensor_list ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.bool_list)
def conveter_aten_ne_bool_list(a: List[bool], b: List[bool], meta_outputs: TensorSpec = None):
    """NB: aten::ne.bool_list(bool[] a, bool[] b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.bool_list ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ne.str_list)
def conveter_aten_ne_str_list(a: List[str], b: List[str], meta_outputs: TensorSpec = None):
    """NB: aten::ne.str_list(str[] a, str[] b) -> bool"""
    raise NotImplementedError("torch.ops.aten.ne.str_list ge_converter is not implemented!")
