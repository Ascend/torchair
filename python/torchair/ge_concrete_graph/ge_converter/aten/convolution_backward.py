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
from torch import Generator, contiguous_format, inf, memory_format, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.convolution_backward.default)
def conveter_aten_convolution_backward_default(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor,
    bias_sizes: Optional[Union[List[int], Tensor]],
    stride: List[int],
    padding: Union[List[int], Tensor],
    dilation: List[int],
    transposed: bool,
    output_padding: Union[List[int], Tensor],
    groups: int,
    output_mask: List[bool],
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None,
):
    """NB: aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.convolution_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.convolution_backward.out)
def conveter_aten_convolution_backward_out(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor,
    bias_sizes: Optional[Union[List[int], Tensor]],
    stride: List[int],
    padding: Union[List[int], Tensor],
    dilation: List[int],
    transposed: bool,
    output_padding: Union[List[int], Tensor],
    groups: int,
    output_mask: List[bool],
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::convolution_backward.out(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise NotImplementedError("torch.ops.aten.convolution_backward.out ge_converter is not implemented!")
