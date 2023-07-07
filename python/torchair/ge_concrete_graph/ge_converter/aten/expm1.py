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


@register_fx_node_ge_converter(torch.ops.aten.expm1.default)
def conveter_aten_expm1_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::expm1(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.expm1.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.expm1.out)
def conveter_aten_expm1_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.expm1.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.expm1.int)
def conveter_aten_expm1_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::expm1.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.expm1.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.expm1.float)
def conveter_aten_expm1_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::expm1.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.expm1.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.expm1.Scalar)
def conveter_aten_expm1_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::expm1.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.expm1.Scalar ge converter is not implement!")


