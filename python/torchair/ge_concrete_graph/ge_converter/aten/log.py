import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
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


@register_fx_node_ge_converter(torch.ops.aten.log.default)
def conveter_aten_log_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.log.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.out)
def conveter_aten_log_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.log.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.int)
def conveter_aten_log_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.log.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.float)
def conveter_aten_log_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.log.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.complex)
def conveter_aten_log_complex(
        a: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.log.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.Scalar)
def conveter_aten_log_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.log.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.int_int)
def conveter_aten_log_int_int(
        a: int,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.int_int(int a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.log.int_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.float_float)
def conveter_aten_log_float_float(
        a: float,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.float_float(float a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.log.float_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.complex_complex)
def conveter_aten_log_complex_complex(
        a: complex,
        b: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.complex_complex(complex a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.log.complex_complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.int_float)
def conveter_aten_log_int_float(
        a: int,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.int_float(int a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.log.int_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.float_int)
def conveter_aten_log_float_int(
        a: float,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.float_int(float a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.log.float_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.int_complex)
def conveter_aten_log_int_complex(
        a: int,
        b: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.int_complex(int a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.log.int_complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.complex_int)
def conveter_aten_log_complex_int(
        a: complex,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.complex_int(complex a, int b) -> complex """
    raise NotImplementedError("torch.ops.aten.log.complex_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.float_complex)
def conveter_aten_log_float_complex(
        a: float,
        b: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.float_complex(float a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.log.float_complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.complex_float)
def conveter_aten_log_complex_float(
        a: complex,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.complex_float(complex a, float b) -> complex """
    raise NotImplementedError("torch.ops.aten.log.complex_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.log.Scalar_Scalar)
def conveter_aten_log_Scalar_Scalar(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::log.Scalar_Scalar(Scalar a, Scalar b) -> float """
    raise NotImplementedError("torch.ops.aten.log.Scalar_Scalar ge converter is not implement!")


