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
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._adaptive_avg_pool2d_backward.default)
def conveter_aten__adaptive_avg_pool2d_backward_default(
    grad_output: Tensor, self: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor"""
    if self.symsize is not None and all([not isinstance(s, torch.SymInt) for s in self.symsize]):
        return ge.AdaptiveAvgPool2dGrad(input_grad=grad_output, orig_input_shape=self.symsize)
    raise NotImplementedError("torch.ops.aten._adaptive_avg_pool2d_backward.default ge_converter is not implemented "
                              "when self is dynamic")


@register_fx_node_ge_converter(torch.ops.aten._adaptive_avg_pool2d_backward.out)
def conveter_aten__adaptive_avg_pool2d_backward_out(
    grad_output: Tensor, self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::_adaptive_avg_pool2d_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten._adaptive_avg_pool2d_backward.out ge_converter is not supported!")
