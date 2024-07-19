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


@register_fx_node_ge_converter(torch.ops.aten._euclidean_dist.default)
def conveter_aten__euclidean_dist_default(
    x1: Tensor, x2: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::_euclidean_dist(Tensor x1, Tensor x2) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._euclidean_dist.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._euclidean_dist.out)
def conveter_aten__euclidean_dist_out(
    x1: Tensor, x2: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::_euclidean_dist.out(Tensor x1, Tensor x2, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._euclidean_dist.out ge_converter is not implemented!")
