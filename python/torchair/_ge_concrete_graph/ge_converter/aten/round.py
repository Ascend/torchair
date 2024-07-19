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
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(32, 1, 128)),
        Support(F16(32, 1, 128)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.round.default)
def conveter_aten_round_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::round(Tensor self) -> Tensor"""
    return ge.Round(self)


@register_fx_node_ge_converter(torch.ops.aten.round.decimals)
def conveter_aten_round_decimals(
    self: Tensor, *, decimals: int, meta_outputs: TensorSpec = None
):
    """NB: aten::round.decimals(Tensor self, *, int decimals) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.round.decimals ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.round.out)
def conveter_aten_round_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.round.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.round.decimals_out)
def conveter_aten_round_decimals_out(
    self: Tensor, *, decimals: int, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::round.decimals_out(Tensor self, *, int decimals, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.round.decimals_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.round.int)
def conveter_aten_round_int(a: int, meta_outputs: TensorSpec = None):
    """NB: aten::round.int(int a) -> float"""
    raise NotImplementedError("torch.ops.aten.round.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.round.float)
def conveter_aten_round_float(a: float, meta_outputs: TensorSpec = None):
    """NB: aten::round.float(float a) -> float"""
    raise NotImplementedError("torch.ops.aten.round.float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.round.Scalar)
def conveter_aten_round_Scalar(a: Union[Number, Tensor], meta_outputs: TensorSpec = None):
    """NB: aten::round.Scalar(Scalar a) -> Scalar"""
    raise NotImplementedError("torch.ops.aten.round.Scalar ge_converter is not implemented!")
