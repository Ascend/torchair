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


@register_fx_node_ge_converter(torch.ops.aten.unsafe_chunk.default)
def conveter_aten_unsafe_chunk_default(
    self: Tensor, chunks: int, dim: int = 0, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.unsafe_chunk.default ge_converter is not implemented!")
