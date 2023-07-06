import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
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
        meta_outputs: Any = None):
    """ NB: aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit.no_stats)
def conveter_aten__native_batch_norm_legit_no_stats(
        input: Tensor,
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        training: bool,
        momentum: float,
        eps: float,
        meta_outputs: Any = None):
    """ NB: aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit.no_stats ge converter is not implement!")


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
        meta_outputs: Any = None):
    """ NB: aten::_native_batch_norm_legit.out(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps, *, Tensor(d!) out, Tensor(e!) save_mean, Tensor(f!) save_invstd) -> (Tensor(d!), Tensor(e!), Tensor(f!)) """
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit.out ge converter is not implement!")


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
        meta_outputs: Any = None):
    """ NB: aten::_native_batch_norm_legit.no_stats_out(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!)) """
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit.no_stats_out ge converter is not implement!")


