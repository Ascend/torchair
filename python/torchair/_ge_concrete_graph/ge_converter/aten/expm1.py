from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.expm1.default)
def conveter_aten_expm1_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::expm1(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.expm1.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.expm1.out)
def conveter_aten_expm1_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.expm1.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.expm1.int)
def conveter_aten_expm1_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::expm1.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.expm1.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.expm1.float)
def conveter_aten_expm1_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::expm1.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.expm1.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.expm1.Scalar)
def conveter_aten_expm1_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::expm1.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.expm1.Scalar ge_converter is not implemented!")
