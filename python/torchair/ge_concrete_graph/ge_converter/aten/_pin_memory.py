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


@register_fx_node_ge_converter(torch.ops.aten._pin_memory.default)
def conveter_aten__pin_memory_default(
        self: Tensor,
        device: Optional[Device] = None,
        meta_outputs: Any = None):
    """ NB: aten::_pin_memory(Tensor self, Device? device=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten._pin_memory.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._pin_memory.out)
def conveter_aten__pin_memory_out(
        self: Tensor,
        device: Optional[Device] = None,
        *,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::_pin_memory.out(Tensor self, Device? device=None, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten._pin_memory.out ge converter is not implement!")


