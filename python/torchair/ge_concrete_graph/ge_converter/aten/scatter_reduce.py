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
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.scatter_reduce.two)
def conveter_aten_scatter_reduce_two(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str,
    *,
    include_self: bool = True,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.scatter_reduce.two ge_converter is not implemented!")


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
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_reduce.two_out(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter_reduce.two_out ge_converter is not implemented!")
