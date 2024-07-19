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


@register_fx_node_ge_converter(torch.ops.aten.meshgrid.default)
def conveter_aten_meshgrid_default(tensors: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::meshgrid(Tensor[] tensors) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.meshgrid.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.meshgrid.indexing)
def conveter_aten_meshgrid_indexing(
    tensors: List[Tensor], *, indexing: str, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.meshgrid.indexing ge_converter is not implemented!")
