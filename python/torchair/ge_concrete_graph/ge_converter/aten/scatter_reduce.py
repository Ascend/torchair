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


@register_fx_node_ge_converter(torch.ops.aten.scatter_reduce.two)
def conveter_aten_scatter_reduce_two(
        self: Tensor,
        dim: int,
        index: Tensor,
        src: Tensor,
        reduce: str,
        *,
        include_self: bool = True,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor """
    raise NotImplementedError("torch.ops.aten.scatter_reduce.two ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.scatter_reduce.two_out)
def conveter_aten_scatter_reduce_two_out(
        self: Tensor,
        dim: int,
        index: Tensor,
        src: Tensor,
        reduce: str,
        *,
        include_self: bool = True,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::scatter_reduce.two_out(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.scatter_reduce.two_out ge converter is not implement!")


