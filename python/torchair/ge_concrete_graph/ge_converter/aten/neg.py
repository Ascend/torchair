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


@register_fx_node_ge_converter(torch.ops.aten.neg.default)
def conveter_aten_neg_default(
        self: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::neg(Tensor self) -> Tensor """
    return ge.Neg(self)


@register_fx_node_ge_converter(torch.ops.aten.neg.out)
def conveter_aten_neg_out(
        self: Tensor,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.neg.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.neg.int)
def conveter_aten_neg_int(
        a: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::neg.int(int a) -> int """
    raise NotImplementedError("torch.ops.aten.neg.int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.neg.float)
def conveter_aten_neg_float(
        a: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::neg.float(float a) -> float """
    raise NotImplementedError("torch.ops.aten.neg.float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.neg.complex)
def conveter_aten_neg_complex(
        a: complex,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::neg.complex(complex a) -> complex """
    raise NotImplementedError("torch.ops.aten.neg.complex ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.neg.Scalar)
def conveter_aten_neg_Scalar(
        a: Union[Number, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::neg.Scalar(Scalar a) -> Scalar """
    raise NotImplementedError("torch.ops.aten.neg.Scalar ge converter is not implement!")


