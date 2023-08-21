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


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.default)
def conveter_aten_logcumsumexp_default(
    self: Tensor, dim: int, meta_outputs: TensorSpec = None
):
    """NB: aten::logcumsumexp(Tensor self, int dim) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logcumsumexp.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.dimname)
def conveter_aten_logcumsumexp_dimname(
    self: Tensor, dim: str, meta_outputs: TensorSpec = None
):
    """NB: aten::logcumsumexp.dimname(Tensor self, str dim) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logcumsumexp.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.dimname_out)
def conveter_aten_logcumsumexp_dimname_out(
    self: Tensor, dim: str, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logcumsumexp.dimname_out(Tensor self, str dim, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logcumsumexp.dimname_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.out)
def conveter_aten_logcumsumexp_out(
    self: Tensor, dim: int, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logcumsumexp.out ge_converter is not implemented!")
