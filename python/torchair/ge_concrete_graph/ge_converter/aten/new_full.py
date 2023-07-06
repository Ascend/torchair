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


@register_fx_node_ge_converter(torch.ops.aten.new_full.default)
def conveter_aten_new_full_default(
        self: Tensor,
        size: Union[List[int], Tensor],
        fill_value: Union[Number, Tensor],
        *,
        dtype: Optional[int] = None,
        layout: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: Optional[bool] = None,
        meta_outputs: Any = None):
    """ NB: aten::new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.new_full.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.new_full.out)
def conveter_aten_new_full_out(
        self: Tensor,
        size: Union[List[int], Tensor],
        fill_value: Union[Number, Tensor],
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::new_full.out(Tensor self, SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.new_full.out ge converter is not implement!")


