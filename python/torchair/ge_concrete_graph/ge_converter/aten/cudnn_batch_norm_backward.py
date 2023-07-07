import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
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


@register_fx_node_ge_converter(torch.ops.aten.cudnn_batch_norm_backward.default)
def conveter_aten_cudnn_batch_norm_backward_default(
        input: Tensor,
        grad_output: Tensor,
        weight: Tensor,
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        save_mean: Optional[Tensor],
        save_var: Optional[Tensor],
        epsilon: float,
        reserveSpace: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace) -> (Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten.cudnn_batch_norm_backward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.cudnn_batch_norm_backward.out)
def conveter_aten_cudnn_batch_norm_backward_out(
        input: Tensor,
        grad_output: Tensor,
        weight: Tensor,
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        save_mean: Optional[Tensor],
        save_var: Optional[Tensor],
        epsilon: float,
        reserveSpace: Tensor,
        *,
        out0: Tensor = None,
        out1: Tensor = None,
        out2: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::cudnn_batch_norm_backward.out(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!)) """
    raise NotImplementedError("torch.ops.aten.cudnn_batch_norm_backward.out ge converter is not implement!")


