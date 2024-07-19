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
from torchair.ge._ge_graph import Tensor, TensorSpec, assert_args_checkout
from torchair._ge_concrete_graph.utils import specific_op_input_layout, specific_op_output_layout


@register_fx_node_ge_converter(torch.ops.aten.max_pool2d_with_indices.default)
def conveter_aten_max_pool2d_with_indices_default(
    self: Tensor,
    kernel_size: List[int],
    stride: List[int] = [],
    padding: List[int] = [0, 0],
    dilation: List[int] = [1, 1],
    ceil_mode: bool = False,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)"""
    """ This converter is a stopgap measure designed to avoid a series issues caused by the imcompatibility between the CANN IR 'MaxPoolWithArgmaxV1' and 
        the aten IR 'max_pool2d_with_indices_backward'. Therefore, no testcast will be set and cannot be set. """
    assert_args_checkout(len(kernel_size) == 1 or len(kernel_size) == 2,
                         "torch.ops.aten.max_pool2d_with_indices.default: kernel_size must either be a single int, "
                         "or a tuple of two ints")
    assert_args_checkout(len(stride) == 0 or len(stride) == 1 or len(stride) == 2,
                         "torch.ops.aten.max_pool2d_with_indices.default: stride must either be omitted, a single "
                         "int, or a tuple of two ints")
    assert_args_checkout(len(padding) == 1 or len(padding) == 2,
                         "torch.ops.aten.max_pool2d_with_indices.default: padding must either be a single int, "
                         "or a tuple of two ints")
    assert_args_checkout(len(dilation) == 1 or len(dilation) == 2,
                         "torch.ops.aten.max_pool2d_with_indices.default: dilation must either be a single int, "
                         "or a tuple of two ints")
    k_h = kernel_size[0]
    k_w = k_h if len(kernel_size) == 1 else kernel_size[1]
    kernel_sizes = [k_h, k_w]
    d_h = k_h if len(stride) == 0 else stride[0]
    d_w = k_w if len(stride) == 0 else d_h if len(stride) == 1 else stride[1]
    strides = [d_h, d_w]
    pad_h = padding[0]
    pad_w = pad_h if len(padding) == 1 else padding[1]
    paddings = [pad_h, pad_w]
    dil_h = dilation[0]
    dil_w = dil_h if len(dilation) == 1 else dilation[1]
    dilation = [dil_h, dil_w]
    ksize = [1, kernel_sizes[0], kernel_sizes[1], 1]
    strides = [1, strides[0], strides[1], 1]
    pads = [1, paddings[0], paddings[1], 1]
    dilations = [1, dilation[0], dilation[1], 1]
    output, argmax = ge.MaxPoolWithArgmaxV1(self, ksize=ksize, strides=strides, \
                                       pads=pads, dilation=dilations, ceil_mode=ceil_mode)
    specific_op_input_layout(output, indices=0, layout="NCHW")
    specific_op_output_layout(output, indices=[0, 1], layout="NCHW")
    argmax = ge.Identity(output)
    return output, argmax


@register_fx_node_ge_converter(torch.ops.aten.max_pool2d_with_indices.out)
def conveter_aten_max_pool2d_with_indices_out(
    self: Tensor,
    kernel_size: List[int],
    stride: List[int] = [],
    padding: List[int] = [0, 0],
    dilation: List[int] = [1, 1],
    ceil_mode: bool = False,
    *,
    out: Tensor = None,
    indices: Tensor = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))"""
    raise NotImplementedError("torch.ops.aten.max_pool2d_with_indices.out ge_converter is not implemented!")
