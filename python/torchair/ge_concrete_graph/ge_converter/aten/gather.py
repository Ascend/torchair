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


@register_fx_node_ge_converter(torch.ops.aten.gather.default)
def conveter_aten_gather_default(
    self: Tensor,
    dim: int,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor"""
    return ge.GatherElements(self, index, dim=dim)


@register_fx_node_ge_converter(torch.ops.aten.gather.out)
def conveter_aten_gather_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gather.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gather.dimname)
def conveter_aten_gather_dimname(
    self: Tensor,
    dim: str,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gather.dimname(Tensor self, str dim, Tensor index, *, bool sparse_grad=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.gather.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.gather.dimname_out)
def conveter_aten_gather_dimname_out(
    self: Tensor,
    dim: str,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::gather.dimname_out(Tensor self, str dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.gather.dimname_out ge_converter is not implemented!")
