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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.unsafe_split_with_sizes.default)
def conveter_aten_unsafe_split_with_sizes_default(
    self: Tensor,
    split_sizes: Union[List[int], Tensor],
    dim: int = 0,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::unsafe_split_with_sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.unsafe_split_with_sizes.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.unsafe_split_with_sizes.out)
def conveter_aten_unsafe_split_with_sizes_out(
    self: Tensor,
    split_sizes: Union[List[int], Tensor],
    dim: int = 0,
    *,
    out: List[Tensor] = None
):
    """NB: aten::unsafe_split_with_sizes.out(Tensor self, SymInt[] split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten.unsafe_split_with_sizes.out ge_converter is not implemented!")
