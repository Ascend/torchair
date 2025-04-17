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


@register_fx_node_ge_converter(torch.ops.aten.unfold_copy.default)
def conveter_aten_unfold_copy_default(
    self: Tensor, dimension: int, size: int, step: int, meta_outputs: TensorSpec = None
):
    """NB: aten::unfold_copy(Tensor self, int dimension, int size, int step) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.unfold_copy.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.unfold_copy.out)
def conveter_aten_unfold_copy_out(
    self: Tensor,
    dimension: int,
    size: int,
    step: int,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::unfold_copy.out(Tensor self, int dimension, int size, int step, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.unfold_copy.out ge_converter is not implemented!")
