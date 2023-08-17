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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.aten.empty.memory_format)
def conveter_aten_empty_memory_format(
    size: Union[List[int], Tensor],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    if layout is not None and (layout != torch.strided):
        raise NotImplementedError("torch.ops.aten.empty.memory_format is only supported on dense tensor now.")

    if memory_format is not None and (memory_format != torch.contiguous_format):
        raise NotImplementedError("torch.ops.aten.empty.memory_format is only supported contiguous_format now.")

    dtype = torch_type_to_ge_type(dtype if dtype is not None else torch.float32)
    return ge.Empty(size, dtype=dtype)

@register_fx_node_ge_converter(torch.ops.aten.empty.out)
def conveter_aten_empty_out(
    size: Union[List[int], Tensor],
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::empty.out(SymInt[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.empty.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.empty.names)
def conveter_aten_empty_names(
    size: List[int],
    names: Optional[List[str]],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::empty.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.empty.names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.empty.names_out)
def conveter_aten_empty_names_out(
    size: List[int],
    names: Optional[List[str]],
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::empty.names_out(int[] size, *, str[]? names, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.empty.names_out ge_converter is not implemented!")
