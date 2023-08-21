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


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit_no_training.default)
def conveter_aten__native_batch_norm_legit_no_training_default(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""
    raise NotImplementedError(
        "torch.ops.aten._native_batch_norm_legit_no_training.default ge_converter is not implemented!"
    )


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit_no_training.out)
def conveter_aten__native_batch_norm_legit_no_training_out(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_native_batch_norm_legit_no_training.out(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise NotImplementedError(
        "torch.ops.aten._native_batch_norm_legit_no_training.out ge_converter is not implemented!"
    )
