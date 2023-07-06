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


@register_fx_node_ge_converter(torch.ops.aten.unsafe_chunk.default)
def conveter_aten_unsafe_chunk_default(
        self: Tensor,
        chunks: int,
        dim: int = 0,
        meta_outputs: Any = None):
    """ NB: aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten.unsafe_chunk.default ge converter is not implement!")


