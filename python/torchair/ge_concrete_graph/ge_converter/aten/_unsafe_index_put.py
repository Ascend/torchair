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


@register_fx_node_ge_converter(torch.ops.aten._unsafe_index_put.default)
def conveter_aten__unsafe_index_put_default(
    self: Tensor,
    indices: List[Optional[Tensor]],
    values: Tensor,
    accumulate: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._unsafe_index_put.default ge_converter is not implemented!")
