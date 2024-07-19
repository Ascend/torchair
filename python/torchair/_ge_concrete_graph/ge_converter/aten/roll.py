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


@register_fx_node_ge_converter(torch.ops.aten.roll.default)
def conveter_aten_roll_default(
    self: Tensor,
    shifts: Union[List[int], Tensor],
    dims: List[int] = [],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::roll(Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.roll.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.roll.out)
def conveter_aten_roll_out(
    self: Tensor,
    shifts: Union[List[int], Tensor],
    dims: List[int] = [],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::roll.out(Tensor self, SymInt[1] shifts, int[1] dims=[], *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.roll.out ge_converter is not implemented!")
