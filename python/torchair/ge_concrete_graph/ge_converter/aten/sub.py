import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten.sub.Tensor)
def conveter_aten_sub_Tensor(
        self: Tensor,
        other: Tensor,
        *,
        alpha: Union[Number, Tensor] = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.sub.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.Scalar)
def conveter_aten_sub_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        alpha: Union[Number, Tensor] = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.sub.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.out)
def conveter_aten_sub_out(
        self: Tensor,
        other: Tensor,
        *,
        alpha: Union[Number, Tensor] = 1,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.sub.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.Scalar_out)
def conveter_aten_sub_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        alpha: Union[Number, Tensor] = 1,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.sub.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.int)
def conveter_aten_sub_int(
        a: int,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.int(int a, int b) -> int """
    raise NotImplementedError("torch.ops.aten.sub.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.complex)
def conveter_aten_sub_complex(
        a: complex,
        b: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.complex(complex a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.sub.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.float)
def conveter_aten_sub_float(
        a: float,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.float(float a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.sub.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.int_complex)
def conveter_aten_sub_int_complex(
        a: int,
        b: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.int_complex(int a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.sub.int_complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.complex_int)
def conveter_aten_sub_complex_int(
        a: complex,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.complex_int(complex a, int b) -> complex """
    raise NotImplementedError("torch.ops.aten.sub.complex_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.float_complex)
def conveter_aten_sub_float_complex(
        a: float,
        b: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.float_complex(float a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.sub.float_complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.complex_float)
def conveter_aten_sub_complex_float(
        a: complex,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.complex_float(complex a, float b) -> complex """
    raise NotImplementedError("torch.ops.aten.sub.complex_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.int_float)
def conveter_aten_sub_int_float(
        a: int,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.int_float(int a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.sub.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.float_int)
def conveter_aten_sub_float_int(
        a: float,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub.float_int(float a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.sub.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sub.default)
def conveter_aten_sub_default(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sub(Scalar a, Scalar b) -> Scalar """
    raise NotImplementedError("torch.ops.aten.sub.default ge converter is not implement!")


