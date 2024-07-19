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


@register_fx_node_ge_converter(torch.ops.aten.to.device)
def conveter_aten_to_device(
    self: Tensor,
    device: Device,
    dtype: int,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.to.device ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.to.dtype)
def conveter_aten_to_dtype(
    self: Tensor,
    dtype: int,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.to.dtype ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.to.other)
def conveter_aten_to_other(
    self: Tensor,
    other: Tensor,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.to.other ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.to.dtype_layout)
def conveter_aten_to_dtype_layout(
    self: Tensor,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.to.dtype_layout ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.to.prim_Device)
def conveter_aten_to_prim_Device(
    self: Tensor,
    device: Optional[Device],
    dtype: Optional[int] = None,
    non_blocking: bool = False,
    copy: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(b|a)"""
    raise NotImplementedError("torch.ops.aten.to.prim_Device ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.to.prim_dtype)
def conveter_aten_to_prim_dtype(
    self: Tensor,
    dtype: Optional[int] = None,
    non_blocking: bool = False,
    copy: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(b|a)"""
    raise NotImplementedError("torch.ops.aten.to.prim_dtype ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.to.prim_other)
def conveter_aten_to_prim_other(
    self: Tensor,
    non_blocking: bool = False,
    copy: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(b|a)"""
    raise NotImplementedError("torch.ops.aten.to.prim_other ge_converter is not implemented!")
