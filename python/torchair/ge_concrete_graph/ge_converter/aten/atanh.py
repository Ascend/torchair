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


@register_fx_node_ge_converter(torch.ops.aten.atanh.default)
def conveter_aten_atanh_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::atanh(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.atanh.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.out)
def conveter_aten_atanh_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.atanh.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.int)
def conveter_aten_atanh_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::atanh.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.atanh.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.float)
def conveter_aten_atanh_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::atanh.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.atanh.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.complex)
def conveter_aten_atanh_complex(
        a: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::atanh.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.atanh.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.atanh.Scalar)
def conveter_aten_atanh_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::atanh.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.atanh.Scalar ge converter is not implement!")


