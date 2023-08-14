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


@register_fx_node_ge_converter(torch.ops.aten._log_softmax_backward_data.default)
def conveter_aten__log_softmax_backward_data_default(
    grad_output: Tensor,
    output: Tensor,
    dim: int,
    input_dtype: int,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._log_softmax_backward_data.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._log_softmax_backward_data.out)
def conveter_aten__log_softmax_backward_data_out(
    grad_output: Tensor,
    output: Tensor,
    dim: int,
    input_dtype: int,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._log_softmax_backward_data.out ge_converter is not implemented!")
