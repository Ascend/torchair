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


@register_fx_node_ge_converter(torch.ops.aten.unbind.int)
def conveter_aten_unbind_int(self: Tensor, dim: int = 0, meta_outputs: List[TensorSpec] = None):
    """NB: aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.unbind.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.unbind.Dimname)
def conveter_aten_unbind_Dimname(self: Tensor, dim: str, meta_outputs: List[TensorSpec] = None):
    """NB: aten::unbind.Dimname(Tensor(a -> *) self, str dim) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.unbind.Dimname ge_converter is not implemented!")
