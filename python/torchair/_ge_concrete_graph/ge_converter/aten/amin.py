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


@register_fx_node_ge_converter(torch.ops.aten.amin.default)
def conveter_aten_amin_default(
    self: Tensor, dim: List[int] = (), keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.amin.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.amin.out)
def conveter_aten_amin_out(
    self: Tensor,
    dim: List[int] = (),
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.amin.out ge_converter is not implemented!")
