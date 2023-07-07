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


@register_fx_node_ge_converter(torch.ops.aten.div.Tensor)
def conveter_aten_div_Tensor(
        self: Tensor,
        other: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.Tensor(Tensor self, Tensor other) -> Tensor """
    return ge.RealDiv(self, other)


@register_fx_node_ge_converter(torch.ops.aten.div.Scalar)
def conveter_aten_div_Scalar(
        self: Tensor,
        other: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.Scalar(Tensor self, Scalar other) -> Tensor """
    raise NotImplementedError("torch.ops.aten.div.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.div.Tensor_mode)
def conveter_aten_div_Tensor_mode(
        self: Tensor,
        other: Tensor,
        *,
        rounding_mode: Optional[str],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor """
    raise NotImplementedError("torch.ops.aten.div.Tensor_mode ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.div.Scalar_mode)
def conveter_aten_div_Scalar_mode(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        rounding_mode: Optional[str],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor """
    raise NotImplementedError("torch.ops.aten.div.Scalar_mode ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.div.out)
def conveter_aten_div_out(
        self: Tensor,
        other: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.div.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.div.out_mode)
def conveter_aten_div_out_mode(
        self: Tensor,
        other: Tensor,
        *,
        rounding_mode: Optional[str],
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.div.out_mode ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.div.Scalar_out)
def conveter_aten_div_Scalar_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.div.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.div.Scalar_mode_out)
def conveter_aten_div_Scalar_mode_out(
        self: Tensor,
        other: Union[Number, Tensor],
        *,
        rounding_mode: Optional[str],
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.Scalar_mode_out(Tensor self, Scalar other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.div.Scalar_mode_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.div.int)
def conveter_aten_div_int(
        a: int,
        b: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.int(int a, int b) -> float """
    raise NotImplementedError("torch.ops.aten.div.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.div.complex)
def conveter_aten_div_complex(
        a: complex,
        b: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.complex(complex a, complex b) -> complex """
    raise NotImplementedError("torch.ops.aten.div.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.div.float)
def conveter_aten_div_float(
        a: float,
        b: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div.float(float a, float b) -> float """
    raise NotImplementedError("torch.ops.aten.div.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.div.default)
def conveter_aten_div_default(
        a: Union[Number, Tensor],
        b: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::div(Scalar a, Scalar b) -> float """
    raise NotImplementedError("torch.ops.aten.div.default ge converter is not implement!")


