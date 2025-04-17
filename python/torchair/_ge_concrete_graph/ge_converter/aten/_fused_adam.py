from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._fused_adam.default)
def conveter_aten__fused_adam_default(
    self: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    amsgrad: bool,
    maximize: bool,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::_fused_adam(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)"""
    raise NotImplementedError("torch.ops.aten._fused_adam.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._fused_adam.out)
def conveter_aten__fused_adam_out(
    self: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    amsgrad: bool,
    maximize: bool,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    out: List[Tensor] = None
):
    """NB: aten::_fused_adam.out(Tensor[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten._fused_adam.out ge_converter is not implemented!")
