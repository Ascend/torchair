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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.binary_cross_entropy.default)
def conveter_aten_binary_cross_entropy_default(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.binary_cross_entropy.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.binary_cross_entropy.out)
def conveter_aten_binary_cross_entropy_out(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.binary_cross_entropy.out ge_converter is not implemented!")
