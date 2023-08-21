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


@register_fx_node_ge_converter(torch.ops.aten.flip.default)
def conveter_aten_flip_default(self: Tensor, dims: List[int], meta_outputs: TensorSpec = None):
    """NB: aten::flip(Tensor self, int[] dims) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.flip.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.flip.out)
def conveter_aten_flip_out(
    self: Tensor, dims: List[int], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::flip.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.flip.out ge_converter is not implemented!")
