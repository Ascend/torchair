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
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.core.utils import logger


@register_fx_node_ge_converter(torch.ops.aten.nll_loss_forward.default)
def conveter_aten_nll_loss_forward_default(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)"""
    reduction_str = ['none', 'mean', 'sum']
    self = dtype_promote(self, target_dtype=meta_outputs[0].dtype)
    if target.dtype == DataType.DT_INT64:
        logger.warning_once("torch.ops.aten.nll_loss_forward.default: "
                            "target shouldn't be the type of int64, it has been changed to int32 automatically.")
        target = dtype_promote(target, target_dtype=DataType.DT_INT32)
    output, total_weight = ge.NLLLoss(self, target, weight, reduction=reduction_str[reduction], ignore_index=ignore_index)
    return output, total_weight


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
    meta_outputs: TensorSpec = None
):
    """NB: aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))"""
    raise RuntimeError("torch.ops.aten.nll_loss_forward.output ge_converter is not supported!")
