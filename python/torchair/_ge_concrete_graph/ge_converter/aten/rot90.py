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


@register_fx_node_ge_converter(torch.ops.aten.rot90.default)
def conveter_aten_rot90_default(
    self: Tensor, k: int = 1, dims: List[int] = [0, 1], meta_outputs: TensorSpec = None
):
    """NB: aten::rot90(Tensor self, int k=1, int[] dims=[0, 1]) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.rot90.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rot90.out)
def conveter_aten_rot90_out(
    self: Tensor,
    k: int = 1,
    dims: List[int] = [0, 1],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rot90.out(Tensor self, int k=1, int[] dims=[0, 1], *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rot90.out ge_converter is not implemented!")
