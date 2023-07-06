import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
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


@register_fx_node_ge_converter(torch.ops.aten._nested_tensor_from_tensor_list.default)
def conveter_aten__nested_tensor_from_tensor_list_default(
        list: List[Tensor],
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Any = None):
    """ NB: aten::_nested_tensor_from_tensor_list(Tensor[] list, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten._nested_tensor_from_tensor_list.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._nested_tensor_from_tensor_list.out)
def conveter_aten__nested_tensor_from_tensor_list_out(
        list: List[Tensor],
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::_nested_tensor_from_tensor_list.out(Tensor[] list, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten._nested_tensor_from_tensor_list.out ge converter is not implement!")


