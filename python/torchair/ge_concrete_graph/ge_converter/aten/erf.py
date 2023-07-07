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


@register_fx_node_ge_converter(torch.ops.aten.erf.default)
def conveter_aten_erf_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::erf(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.erf.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.erf.out)
def conveter_aten_erf_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.erf.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.erf.int)
def conveter_aten_erf_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::erf.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.erf.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.erf.float)
def conveter_aten_erf_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::erf.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.erf.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.erf.Scalar)
def conveter_aten_erf_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::erf.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.erf.Scalar ge converter is not implement!")


