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


@register_fx_node_ge_converter(torch.ops.aten.index_reduce.default)
def conveter_aten_index_reduce_default(
        self: Tensor,
        dim: int,
        index: Tensor,
        source: Tensor,
        reduce: str,
        *,
        include_self: bool = True,
        meta_outputs: Any = None):
    """ NB: aten::index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor """
    raise NotImplementedError("torch.ops.aten.index_reduce.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index_reduce.out)
def conveter_aten_index_reduce_out(
        self: Tensor,
        dim: int,
        index: Tensor,
        source: Tensor,
        reduce: str,
        *,
        include_self: bool = True,
        out: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::index_reduce.out(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.index_reduce.out ge converter is not implement!")


