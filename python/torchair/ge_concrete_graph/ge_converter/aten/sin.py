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


@register_fx_node_ge_converter(torch.ops.aten.sin.default)
def conveter_aten_sin_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sin(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.sin.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sin.out)
def conveter_aten_sin_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.sin.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sin.int)
def conveter_aten_sin_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sin.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.sin.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sin.float)
def conveter_aten_sin_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sin.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.sin.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sin.complex)
def conveter_aten_sin_complex(
        a: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sin.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.sin.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sin.Scalar)
def conveter_aten_sin_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::sin.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.sin.Scalar ge converter is not implement!")


