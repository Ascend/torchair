import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided
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


@register_fx_node_ge_converter(torch.ops.aten._native_batch_norm_legit_no_training.default)
def conveter_aten__native_batch_norm_legit_no_training_default(
        input: Tensor,
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        running_mean: Tensor,
        running_var: Tensor,
        momentum: float,
        eps: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit_no_training.default ge converter is not implement!")


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
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_native_batch_norm_legit_no_training.out(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!)) """
    raise NotImplementedError("torch.ops.aten._native_batch_norm_legit_no_training.out ge converter is not implement!")


