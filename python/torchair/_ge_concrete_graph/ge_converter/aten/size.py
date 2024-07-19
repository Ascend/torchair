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


@register_fx_node_ge_converter(torch.ops.aten.size.int)
def conveter_aten_size_int(self: Tensor, dim: int, meta_outputs: TensorSpec = None):
    """NB: aten::size.int(Tensor self, int dim) -> int"""
    raise NotImplementedError("torch.ops.aten.size.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.size.Dimname)
def conveter_aten_size_Dimname(self: Tensor, dim: str, meta_outputs: TensorSpec = None):
    """NB: aten::size.Dimname(Tensor self, str dim) -> int"""
    raise NotImplementedError("torch.ops.aten.size.Dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.size.default)
def conveter_aten_size_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::size(Tensor self) -> int[]"""
    raise NotImplementedError("torch.ops.aten.size.default ge_converter is not implemented!")
