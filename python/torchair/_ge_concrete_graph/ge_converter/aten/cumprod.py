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


@register_fx_node_ge_converter(torch.ops.aten.cumprod.default)
def conveter_aten_cumprod_default(
    self: Tensor, dim: int, *, dtype: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.cumprod.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod.dimname)
def conveter_aten_cumprod_dimname(
    self: Tensor, dim: str, *, dtype: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::cumprod.dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.cumprod.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod.dimname_out)
def conveter_aten_cumprod_dimname_out(
    self: Tensor,
    dim: str,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cumprod.dimname_out(Tensor self, str dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cumprod.dimname_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.cumprod.out)
def conveter_aten_cumprod_out(
    self: Tensor,
    dim: int,
    *,
    dtype: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.cumprod.out ge_converter is not implemented!")
