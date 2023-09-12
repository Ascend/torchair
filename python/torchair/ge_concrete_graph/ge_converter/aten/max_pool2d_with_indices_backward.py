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
from torchair.ge_concrete_graph.utils import specific_op_input_layout, specific_op_output_layout


@register_fx_node_ge_converter(torch.ops.aten.max_pool2d_with_indices_backward.default)
def conveter_aten_max_pool2d_with_indices_backward_default(
    grad_output: Tensor,
    self: Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool,
    indices: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor"""
    """ This converter is a stopgap measure designed to avoid a series issues caused by the imcompatibility between the CANN IR 'MaxPoolWithArgmaxV1' and 
        the aten IR 'max_pool2d_with_indices_backward'. Therefore, no testcast will be set and cannot be set. """
    ksize = [1, kernel_size[0], kernel_size[1], 1]
    strides = [1, stride[0], stride[1], 1]
    pads = [1, padding[0], padding[1], 1]
    dilations = [1, dilation[0], dilation[1], 1]
    output, argmax = ge.MaxPoolWithArgmaxV1(self, ksize=ksize, \
                                            strides=strides, pads=pads, dilation=dilations, ceil_mode=ceil_mode)
    specific_op_input_layout(output, indices=0, layout="NCHW")
    specific_op_output_layout(output, indices=[0, 1], layout="NCHW")
    grad_input = ge.MaxPoolGradWithArgmaxV1(self, grad_output, argmax, ksize=ksize, \
                                            strides=strides, pads=pads, dilation=dilations, ceil_mode=ceil_mode)
    specific_op_input_layout(grad_input, indices=[0, 1, 2], layout="NCHW")
    specific_op_output_layout(grad_input, indices=0, layout="NCHW")
    return grad_input


@register_fx_node_ge_converter(torch.ops.aten.max_pool2d_with_indices_backward.grad_input)
def conveter_aten_max_pool2d_with_indices_backward_grad_input(
    grad_output: Tensor,
    self: Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool,
    indices: Tensor,
    *,
    grad_input: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise NotImplementedError(
        "torch.ops.aten.max_pool2d_with_indices_backward.grad_input ge_converter is not implemented!"
    )
