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


@register_fx_node_ge_converter(torch.ops.aten.nll_loss_forward.default)
def conveter_aten_nll_loss_forward_default(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: Union[int, Tensor],
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)"""
    raise NotImplementedError("torch.ops.aten.nll_loss_forward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.nll_loss_forward.output)
def conveter_aten_nll_loss_forward_output(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: Union[int, Tensor],
    *,
    output: Tensor = None,
    total_weight: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))"""
    raise NotImplementedError("torch.ops.aten.nll_loss_forward.output ge_converter is not implemented!")
