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


@register_fx_node_ge_converter(torch.ops.aten.unsafe_split_with_sizes.default)
def conveter_aten_unsafe_split_with_sizes_default(
        self: Tensor,
        split_sizes: Union[List[int], Tensor],
        dim: int = 0,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::unsafe_split_with_sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten.unsafe_split_with_sizes.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.unsafe_split_with_sizes.out)
def conveter_aten_unsafe_split_with_sizes_out(
        self: Tensor,
        split_sizes: Union[List[int], Tensor],
        dim: int = 0,
        *,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::unsafe_split_with_sizes.out(Tensor self, SymInt[] split_sizes, int dim=0, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten.unsafe_split_with_sizes.out ge converter is not implement!")


