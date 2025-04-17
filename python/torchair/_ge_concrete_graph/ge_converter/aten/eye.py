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


@register_fx_node_ge_converter(torch.ops.aten.eye.default)
def conveter_aten_eye_default(
    n: Union[int, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::eye(SymInt n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.eye.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.eye.m)
def conveter_aten_eye_m(
    n: Union[int, Tensor],
    m: Union[int, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::eye.m(SymInt n, SymInt m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.eye.m ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.eye.out)
def conveter_aten_eye_out(
    n: Union[int, Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::eye.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.eye.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.eye.m_out)
def conveter_aten_eye_m_out(
    n: Union[int, Tensor],
    m: Union[int, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::eye.m_out(SymInt n, SymInt m, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.eye.m_out ge_converter is not implemented!")
