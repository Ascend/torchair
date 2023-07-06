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


@register_fx_node_ge_converter(torch.ops.aten.clone.default)
def conveter_aten_clone_default(
        self: Tensor,
        *,
        memory_format: Optional[int] = None,
        meta_outputs: Any = None):
    """ NB: aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor """
    if memory_format is not None and memory_format is not torch.contiguous_format:
        # TODO: fix this case
        print(f"[warning] torch.ops.aten.clone.default have some unprocessed parameters or cases!")

    return ge.Identity(self)


@register_fx_node_ge_converter(torch.ops.aten.clone.out)
def conveter_aten_clone_out(
        self: Tensor,
        *,
        memory_format: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::clone.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.clone.out ge converter is not implement!")


