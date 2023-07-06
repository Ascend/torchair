import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, torch_type_to_ge_type
from torchair.ge_concrete_graph.ge_graph import Tensor
from torchair.ge_concrete_graph.utils import dtype_promote
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


@register_fx_node_ge_converter(torch.ops.aten.arange.default)
def conveter_aten_arange_default(
        end: Union[Number, Tensor],
        *,
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Any = None):
    """ NB: aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    target_dtype = dtype if dtype is not None else meta_outputs.dtype
    start, limit, delta = dtype_promote(0, end, 1, target_dtype = target_dtype)

    # layout, pin_memory and device have no effect on constructing graph.
    return ge.Range(start, limit, delta)


@register_fx_node_ge_converter(torch.ops.aten.arange.start)
def conveter_aten_arange_start(
        start: Union[Number, Tensor],
        end: Union[Number, Tensor],
        *,
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Any = None):
    """ NB: aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.arange.start ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.arange.start_step)
def conveter_aten_arange_start_step(
        start: Union[Number, Tensor],
        end: Union[Number, Tensor],
        step: Union[Number, Tensor] = 1,
        *,
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Any = None):
    """ NB: aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.arange.start_step ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.arange.start_out)
def conveter_aten_arange_start_out(
        start: Union[Number, Tensor],
        end: Union[Number, Tensor],
        step: Union[Number, Tensor] = 1,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.arange.start_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.arange.out)
def conveter_aten_arange_out(
        end: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.arange.out ge converter is not implement!")


