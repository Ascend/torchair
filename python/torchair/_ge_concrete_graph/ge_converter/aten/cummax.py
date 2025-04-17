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


@register_fx_node_ge_converter(torch.ops.aten.cummax.default)
def conveter_aten_cummax_default(self: Tensor, dim: int, meta_outputs: TensorSpec = None):
    """NB: aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)"""
    raise NotImplementedError("torch.ops.aten.cummax.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cummax.dimname)
def conveter_aten_cummax_dimname(self: Tensor, dim: str, meta_outputs: TensorSpec = None):
    """NB: aten::cummax.dimname(Tensor self, str dim) -> (Tensor values, Tensor indices)"""
    raise NotImplementedError("torch.ops.aten.cummax.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cummax.dimname_out)
def conveter_aten_cummax_dimname_out(
    self: Tensor,
    dim: str,
    *,
    values: Tensor = None,
    indices: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cummax.dimname_out(Tensor self, str dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise NotImplementedError("torch.ops.aten.cummax.dimname_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cummax.out)
def conveter_aten_cummax_out(
    self: Tensor,
    dim: int,
    *,
    values: Tensor = None,
    indices: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)"""
    raise NotImplementedError("torch.ops.aten.cummax.out ge_converter is not implemented!")
