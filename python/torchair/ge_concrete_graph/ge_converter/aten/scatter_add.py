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


@register_fx_node_ge_converter(torch.ops.aten.scatter_add.default)
def conveter_aten_scatter_add_default(
    self: Tensor, dim: int, index: Tensor, src: Tensor, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.scatter_add.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter_add.out)
def conveter_aten_scatter_add_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scatter_add.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.scatter_add.dimname)
def conveter_aten_scatter_add_dimname(
    self: Tensor, dim: str, index: Tensor, src: Tensor, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::scatter_add.dimname(Tensor self, str dim, Tensor index, Tensor src) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.scatter_add.dimname ge_converter is not implemented!")
