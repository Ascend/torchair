from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.lgamma.default)
def conveter_aten_lgamma_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::lgamma(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.lgamma.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.out)
def conveter_aten_lgamma_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.lgamma.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.int)
def conveter_aten_lgamma_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::lgamma.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.lgamma.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.float)
def conveter_aten_lgamma_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::lgamma.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.lgamma.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lgamma.Scalar)
def conveter_aten_lgamma_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::lgamma.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.lgamma.Scalar ge_converter is not implemented!")
