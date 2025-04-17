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


@register_fx_node_ge_converter(torch.ops.aten.normal.Tensor_float)
def conveter_aten_normal_Tensor_float(
    mean: Tensor,
    std: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::normal.Tensor_float(Tensor mean, float std=1., *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.normal.Tensor_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.normal.Tensor_float_out)
def conveter_aten_normal_Tensor_float_out(
    mean: Tensor,
    std: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::normal.Tensor_float_out(Tensor mean, float std=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.normal.Tensor_float_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.normal.float_Tensor_out)
def conveter_aten_normal_float_Tensor_out(
    mean: float,
    std: Tensor,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.normal.float_Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.normal.float_Tensor)
def conveter_aten_normal_float_Tensor(
    mean: float,
    std: Tensor,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.normal.float_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.normal.Tensor_Tensor)
def conveter_aten_normal_Tensor_Tensor(
    mean: Tensor,
    std: Tensor,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.normal.Tensor_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.normal.Tensor_Tensor_out)
def conveter_aten_normal_Tensor_Tensor_out(
    mean: Tensor,
    std: Tensor,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.normal.Tensor_Tensor_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.normal.float_float)
def conveter_aten_normal_float_float(
    mean: float,
    std: float,
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator] = None,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.normal.float_float ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.normal.float_float_out)
def conveter_aten_normal_float_float_out(
    mean: float,
    std: float,
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::normal.float_float_out(float mean, float std, SymInt[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.normal.float_float_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.normal.out)
def conveter_aten_normal_out(
    self: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::normal.out(Tensor self, float mean=0., float std=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.normal.out ge_converter is not implemented!")
