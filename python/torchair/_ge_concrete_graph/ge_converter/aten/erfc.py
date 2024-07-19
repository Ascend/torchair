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


@register_fx_node_ge_converter(torch.ops.aten.erfc.default)
def conveter_aten_erfc_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::erfc(Tensor self) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.erfc.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.erfc.out)
def conveter_aten_erfc_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.erfc.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.erfc.int)
def conveter_aten_erfc_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::erfc.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.erfc.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.erfc.float)
def conveter_aten_erfc_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::erfc.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.erfc.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.erfc.Scalar)
def conveter_aten_erfc_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::erfc.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.erfc.Scalar ge_converter is not implemented!")
