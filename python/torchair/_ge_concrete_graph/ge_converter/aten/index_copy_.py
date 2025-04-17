from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.index_copy_.default)
def conveter_aten_index_copy__default(
    self: Tensor, dim: int, index: Tensor, source: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_copy_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_copy_.dimname)
def conveter_aten_index_copy__dimname(
    self: Tensor, dim: str, index: Tensor, source: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::index_copy_.dimname(Tensor(a!) self, str dim, Tensor index, Tensor source) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_copy_.dimname ge_converter is not implemented!")
