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


@register_fx_node_ge_converter(torch.ops.aten.sqrt.default)
def conveter_aten_sqrt_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sqrt(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.sqrt.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sqrt.out)
def conveter_aten_sqrt_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.sqrt.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sqrt.int)
def conveter_aten_sqrt_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sqrt.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.sqrt.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sqrt.float)
def conveter_aten_sqrt_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sqrt.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.sqrt.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sqrt.complex)
def conveter_aten_sqrt_complex(
        a: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sqrt.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.sqrt.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sqrt.Scalar)
def conveter_aten_sqrt_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sqrt.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.sqrt.Scalar ge converter is not implement!")


