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
from torch import Generator, contiguous_format, inf, memory_format, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.mse_loss.default)
def conveter_aten_mse_loss_default(
    self: Tensor, target: Tensor, reduction: int = 1, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::mse_loss(Tensor self, Tensor target, int reduction=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.mse_loss.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.mse_loss.out)
def conveter_aten_mse_loss_out(
    self: Tensor,
    target: Tensor,
    reduction: int = 1,
    *,
    out: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::mse_loss.out(Tensor self, Tensor target, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.mse_loss.out ge_converter is not implemented!")
