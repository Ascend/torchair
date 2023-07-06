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


@register_fx_node_ge_converter(torch.ops.aten.cudnn_batch_norm.default)
def conveter_aten_cudnn_batch_norm_default(
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        training: bool,
        exponential_average_factor: float,
        epsilon: float,
        meta_outputs: Any = None):
    """ NB: aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten.cudnn_batch_norm.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cudnn_batch_norm.out)
def conveter_aten_cudnn_batch_norm_out(
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        training: bool,
        exponential_average_factor: float,
        epsilon: float,
        *,
        out0: Tensor = None,
        out1: Tensor = None,
        out2: Tensor = None,
        out3: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::cudnn_batch_norm.out(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!)) """
    raise NotImplementedError("torch.ops.aten.cudnn_batch_norm.out ge converter is not implement!")


