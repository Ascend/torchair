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


@register_fx_node_ge_converter(torch.ops.aten.renorm.default)
def conveter_aten_renorm_default(
    self: Tensor,
    p: Union[Number, Tensor],
    dim: int,
    maxnorm: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.renorm.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.renorm.out)
def conveter_aten_renorm_out(
    self: Tensor,
    p: Union[Number, Tensor],
    dim: int,
    maxnorm: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.renorm.out ge_converter is not implemented!")
