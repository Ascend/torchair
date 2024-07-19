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


@register_fx_node_ge_converter(torch.ops.aten.nll_loss.default)
def conveter_aten_nll_loss_default(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = 1,
    ignore_index: Union[int, Tensor] = -100,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, SymInt ignore_index=-100) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.nll_loss.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.nll_loss.out)
def conveter_aten_nll_loss_out(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = 1,
    ignore_index: Union[int, Tensor] = -100,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, SymInt ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.nll_loss.out ge_converter is not implemented!")
