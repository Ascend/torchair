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


@register_fx_node_ge_converter(torch.ops.aten.split.Tensor)
def conveter_aten_split_Tensor(
    self: Tensor, split_size: Union[int, Tensor], dim: int = 0, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]"""
    split_sizes = split_size
    if isinstance(split_sizes, int):
        split_sizes = [split_size for _ in range(len(meta_outputs))]
        split_sizes[-1] = -1
    return ge.SplitV(self, split_sizes, dim, num_split=len(split_sizes))


@register_fx_node_ge_converter(torch.ops.aten.split.sizes)
def conveter_aten_split_sizes(
    self: Tensor,
    split_size: Union[List[int], Tensor],
    dim: int = 0,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::split.sizes(Tensor(a -> *) self, SymInt[] split_size, int dim=0) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.split.sizes ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.split.str)
def conveter_aten_split_str(
    self: str, separator: Optional[str] = None, max: int = -1, meta_outputs: TensorSpec = None
):
    """NB: aten::split.str(str self, str? separator=None, int max=-1) -> str[]"""
    raise NotImplementedError("torch.ops.aten.split.str ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.split.default)
def conveter_aten_split_default(
    self: Tensor, split_sizes: List[int], dim: int = 0, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::split(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.split.default ge_converter is not implemented!")
