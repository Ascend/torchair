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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote

@register_fx_node_ge_converter(torch.ops.aten.zeros.names)
def conveter_aten_zeros_names(
    size: List[int],
    *,
    names: Optional[List[str]],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::zeros.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise RuntimeError("torch.ops.aten.zeros.names ge_converter is not supported!")

@declare_supported(
    [
        Support((2, 3)),
        Support((2, 3), dtype=torch.int),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.zeros.default)
def conveter_aten_zeros_default(
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        return ge.Fill(size, 0.)
    else:
        return ge.Fill(size, ge.Cast(0., dst_type=torch_type_to_ge_type(dtype)))


@register_fx_node_ge_converter(torch.ops.aten.zeros.names_out)
def conveter_aten_zeros_names_out(
    size: List[int],
    *,
    names: Optional[List[str]],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::zeros.names_out(int[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.zeros.names_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.zeros.out)
def conveter_aten_zeros_out(
    size: Union[List[int], Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.zeros.out ge_converter is not supported!")
