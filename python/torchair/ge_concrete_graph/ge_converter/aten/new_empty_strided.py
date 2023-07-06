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


@register_fx_node_ge_converter(torch.ops.aten.new_empty_strided.default)
def conveter_aten_new_empty_strided_default(
        self: Tensor,
        size: Union[List[int], Tensor],
        stride: Union[List[int], Tensor],
        *,
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Any = None):
    """ NB: aten::new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.new_empty_strided.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.new_empty_strided.out)
def conveter_aten_new_empty_strided_out(
        self: Tensor,
        size: Union[List[int], Tensor],
        stride: Union[List[int], Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::new_empty_strided.out(Tensor self, SymInt[] size, SymInt[] stride, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.new_empty_strided.out ge converter is not implement!")


