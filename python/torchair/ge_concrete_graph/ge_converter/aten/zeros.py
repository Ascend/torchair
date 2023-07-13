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


@register_fx_node_ge_converter(torch.ops.aten.zeros.names)
def conveter_aten_zeros_names(
        size: List[int],
        *,
        names: Optional[List[str]],
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::zeros.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.zeros.names ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.zeros.default)
def conveter_aten_zeros_default(
        size: Union[List[int], Tensor],
        *,
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.zeros.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.zeros.names_out)
def conveter_aten_zeros_names_out(
        size: List[int],
        *,
        names: Optional[List[str]],
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::zeros.names_out(int[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.zeros.names_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.zeros.out)
def conveter_aten_zeros_out(
        size: Union[List[int], Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.zeros.out ge converter is not implement!")


