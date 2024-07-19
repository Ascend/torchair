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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, torch_type_to_ge_type
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.ones.names)
def conveter_aten_ones_names(
    size: List[int],
    *,
    names: Optional[List[str]],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::ones.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.ones.names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ones.default)
def conveter_aten_ones_default(
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::ones(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    # ScalarType default is float, reference by c10/core/DefaultDtype.cpp
    dtype = torch_type_to_ge_type(dtype) if dtype is not None else DataType.DT_FLOAT
    value = ge.Const(1, dtype=dtype)
    return ge.Fill(size, value)


@register_fx_node_ge_converter(torch.ops.aten.ones.names_out)
def conveter_aten_ones_names_out(
    size: List[int],
    *,
    names: Optional[List[str]],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::ones.names_out(int[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.ones.names_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.ones.out)
def conveter_aten_ones_out(
    size: Union[List[int], Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::ones.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.ones.out ge_converter is not implemented!")
