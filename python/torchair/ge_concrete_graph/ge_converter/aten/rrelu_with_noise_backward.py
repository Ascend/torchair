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


@register_fx_node_ge_converter(torch.ops.aten.rrelu_with_noise_backward.default)
def conveter_aten_rrelu_with_noise_backward_default(
        grad_output: Tensor,
        self: Tensor,
        noise: Tensor,
        lower: Union[Number, Tensor],
        upper: Union[Number, Tensor],
        training: bool,
        self_is_result: bool,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor """
    raise NotImplementedError("torch.ops.aten.rrelu_with_noise_backward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.rrelu_with_noise_backward.out)
def conveter_aten_rrelu_with_noise_backward_out(
        grad_output: Tensor,
        self: Tensor,
        noise: Tensor,
        lower: Union[Number, Tensor],
        upper: Union[Number, Tensor],
        training: bool,
        self_is_result: bool,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::rrelu_with_noise_backward.out(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.rrelu_with_noise_backward.out ge converter is not implement!")


