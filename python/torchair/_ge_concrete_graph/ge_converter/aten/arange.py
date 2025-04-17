from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(100, dtype=torch.bfloat16),
    Support(100, dtype=torch.int32),
    Support(100, dtype=torch.float16)
])
@register_fx_node_ge_converter(torch.ops.aten.arange.default)
def conveter_aten_arange_default(
    end: Union[Number, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    target_dtype = dtype if dtype else meta_outputs.dtype
    start = ge.Const(0, DataType.DT_INT32)
    step = ge.Const(1, DataType.DT_INT32)
    result = dtype_promote(ge.Range(start, end, step), target_dtype=target_dtype)

    # layout, pin_memory and device have no effect on constructing graph.
    return result


@declare_supported([
    Support(0, 100, dtype=torch.bfloat16),
    Support(0, 100, dtype=torch.int32),
    Support(0, 100, dtype=torch.float16),
    Support(2, 100, dtype=torch.float16)
])
@register_fx_node_ge_converter(torch.ops.aten.arange.start)
def conveter_aten_arange_start(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    target_dtype = dtype if dtype else meta_outputs.dtype
    step = ge.Const(1, DataType.DT_INT32)
    result = dtype_promote(ge.Range(start, end, step), target_dtype=target_dtype)

    # layout, pin_memory and device have no effect on constructing graph.
    return result


@declare_supported([
    Support(0, 100, 1, dtype=torch.bfloat16),
    Support(0, 100, 1, dtype=torch.int32),
    Support(0, 100, 2, dtype=torch.int32),
    Support(0, 100, 2, dtype=torch.float16),
    Support(0, 100, 2.5, dtype=torch.float16),
    Support(0, 100, 2.5),
])
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
    meta_outputs: TensorSpec = None
):
    """NB: aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    target_dtype = dtype if dtype else meta_outputs.dtype
    start, end, step = dtype_promote(start, end, step, target_dtype=target_dtype)
    result = dtype_promote(ge.Range(start, end, step), target_dtype=target_dtype)

    # layout, pin_memory and device have no effect on constructing graph.
    return result


@register_fx_node_ge_converter(torch.ops.aten.arange.start_out)
def conveter_aten_arange_start_out(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    step: Union[Number, Tensor] = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.arange.start_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.arange.out)
def conveter_aten_arange_out(
    end: Union[Number, Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.arange.out ge_converter is not supported!")
