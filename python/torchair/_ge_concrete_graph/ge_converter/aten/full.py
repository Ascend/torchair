from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import numpy as np
import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.aten.full.names)
def conveter_aten_full_names(
    size: List[int],
    fill_value: Union[Number, Tensor],
    *,
    names: Optional[List[str]],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::full.names(int[] size, Scalar fill_value, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.full.names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.full.default)
def conveter_aten_full_default(
    size: Union[List[int], Tensor],
    fill_value: Union[Number, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    target_dtype = dtype if dtype is not None else meta_outputs.dtype
    fill_value = dtype_promote(fill_value, target_dtype=target_dtype)

    # layout, pin_memory and device have no effect on constructing graph.
    if not isinstance(size, Tensor) and np.array(size).size == 0:
        return fill_value
    else:
        return ge.Fill(size, fill_value)


@register_fx_node_ge_converter(torch.ops.aten.full.names_out)
def conveter_aten_full_names_out(
    size: List[int],
    fill_value: Union[Number, Tensor],
    *,
    names: Optional[List[str]],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::full.names_out(int[] size, Scalar fill_value, *, str[]? names, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.full.names_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.full.out)
def conveter_aten_full_out(
    size: Union[List[int], Tensor],
    fill_value: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::full.out(SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.full.out ge_converter is not implemented!")
