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


@register_fx_node_ge_converter(torch.ops.aten.to.device)
def conveter_aten_to_device(
        self: Tensor,
        device: Device,
        dtype: int,
        non_blocking: bool = False,
        copy: bool = False,
        memory_format: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.to.device ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.to.dtype)
def conveter_aten_to_dtype(
        self: Tensor,
        dtype: int,
        non_blocking: bool = False,
        copy: bool = False,
        memory_format: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.to.dtype ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.to.other)
def conveter_aten_to_other(
        self: Tensor,
        other: Tensor,
        non_blocking: bool = False,
        copy: bool = False,
        memory_format: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.to.other ge converter is not implement!")


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
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.to.dtype_layout ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.to.prim_Device)
def conveter_aten_to_prim_Device(
        self: Tensor,
        device: Optional[Device],
        dtype: Optional[int] = None,
        non_blocking: bool = False,
        copy: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(b|a) """
    raise NotImplementedError("torch.ops.aten.to.prim_Device ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.to.prim_dtype)
def conveter_aten_to_prim_dtype(
        self: Tensor,
        dtype: Optional[int] = None,
        non_blocking: bool = False,
        copy: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(b|a) """
    raise NotImplementedError("torch.ops.aten.to.prim_dtype ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.to.prim_other)
def conveter_aten_to_prim_other(
        self: Tensor,
        non_blocking: bool = False,
        copy: bool = False,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(b|a) """
    raise NotImplementedError("torch.ops.aten.to.prim_other ge converter is not implement!")


