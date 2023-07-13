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


@register_fx_node_ge_converter(torch.ops.aten.exp.default)
def conveter_aten_exp_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::exp(Tensor self) -> Tensor """
    raise NotImplementedError("torch.ops.aten.exp.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.exp.out)
def conveter_aten_exp_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.exp.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.exp.int)
def conveter_aten_exp_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::exp.int(int a) -> float """
    raise NotImplementedError("torch.ops.aten.exp.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.exp.float)
def conveter_aten_exp_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::exp.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.exp.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.exp.complex)
def conveter_aten_exp_complex(
        a: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::exp.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.exp.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.exp.Scalar)
def conveter_aten_exp_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::exp.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.exp.Scalar ge converter is not implement!")


