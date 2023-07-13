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


@register_fx_node_ge_converter(torch.ops.aten.meshgrid.default)
def conveter_aten_meshgrid_default(
        tensors: List[Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::meshgrid(Tensor[] tensors) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten.meshgrid.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.meshgrid.indexing)
def conveter_aten_meshgrid_indexing(
        tensors: List[Tensor],
        *,
        indexing: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten.meshgrid.indexing ge converter is not implement!")


