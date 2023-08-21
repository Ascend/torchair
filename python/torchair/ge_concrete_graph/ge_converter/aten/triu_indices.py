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


@register_fx_node_ge_converter(torch.ops.aten.triu_indices.default)
def conveter_aten_triu_indices_default(
    row: int,
    col: int,
    offset: int = 0,
    *,
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.triu_indices.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.triu_indices.out)
def conveter_aten_triu_indices_out(
    row: int, col: int, offset: int = 0, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::triu_indices.out(int row, int col, int offset=0, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.triu_indices.out ge_converter is not implemented!")
