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


@register_fx_node_ge_converter(torch.ops.aten.split_with_sizes.default)
def conveter_aten_split_with_sizes_default(
    self: Tensor,
    split_sizes: Union[List[int], Tensor],
    dim: int = 0,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.split_with_sizes.default ge_converter is not implemented!")
