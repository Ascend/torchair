from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.utils import specific_op_input_layout, \
    specific_op_output_layout
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported(
    [
        Support(F32(2, 2, 2), F32(2), F32(2), F32(2), F32(2), training=True, momentum=0.9, eps=1e-5),
        Support(F32(2, 2, 2, 2), F32(2), F32(2), F32(2), F32(2), training=True, momentum=0.9, eps=1e-5),
        Support(F32(2, 2, 2, 2, 2), F32(2), F32(2), F32(2), F32(2), training=True, momentum=0.9, eps=1e-5),
        Support(F32(2, 2, 2, 2), None, None, F32(2), F32(2), training=True, momentum=0.9, eps=1e-5),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit.default)
def conveter_aten__native_batch_norm_legit_default(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit.no_stats)
def conveter_aten__native_batch_norm_legit_no_stats(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit.no_stats ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit.out)
def conveter_aten__native_batch_norm_legit_out(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
    *,
    out: Tensor = None,
    save_mean: Tensor = None,
    save_invstd: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_native_batch_norm_legit.out(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps, *, Tensor(d!) out, Tensor(e!) save_mean, Tensor(f!) save_invstd) -> (Tensor(d!), Tensor(e!), Tensor(f!))"""
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit.no_stats_out)
def conveter_aten__native_batch_norm_legit_no_stats_out(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
    *,
    out: Tensor = None,
    save_mean: Tensor = None,
    save_invstd: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_native_batch_norm_legit.no_stats_out(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit.no_stats_out ge_converter is not implemented!")
