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


@register_fx_node_ge_converter(torch.ops.aten.logspace.default)
def conveter_aten_logspace_default(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    steps: int,
    base: float = 10.0,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::logspace(Scalar start, Scalar end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logspace.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logspace.out)
def conveter_aten_logspace_out(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    steps: int,
    base: float = 10.0,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::logspace.out(Scalar start, Scalar end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logspace.out ge_converter is not implemented!")
