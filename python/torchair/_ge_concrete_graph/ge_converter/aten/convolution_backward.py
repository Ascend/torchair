from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, \
    specific_op_output_layout
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support
from torchair.core.utils import logger


def conv3d_backward_input_nocheck(x: Tensor, grad: Tensor, weight: Tensor, stride: List[int], padding: List[int],
                                  dilation: List[int], groups: int):
    strides = [1, 1, stride[0], stride[1], stride[2]]
    pads = [padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]]
    dilation = [1, 1, dilation[0], dilation[1], dilation[2]]

    output = ge.Conv3DBackpropInput(input_size=ge.Shape(x), filter=weight, out_backprop=grad,
                                    strides=strides, pads=pads, dilations=dilation, groups=groups, data_format="NCDHW")

    specific_op_input_layout(output, indices=0, layout="ND")
    specific_op_input_layout(output, indices=[1, 2], layout="NCDHW")
    specific_op_output_layout(output, indices=0, layout="NCDHW")
    return output


def conv3d_backward_weight_nocheck(x: Tensor, grad: Tensor, weight: Tensor, stride: List[int], padding: List[int],
                                   dilation: List[int], groups: int):
    strides = [1, 1, stride[0], stride[1], stride[2]]
    pads = [padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]]
    dilation = [1, 1, dilation[0], dilation[1], dilation[2]]
    output = ge.Conv3DBackpropFilter(x=x, filter_size=ge.Shape(weight), out_backprop=grad,
                                     strides=strides, pads=pads, dilations=dilation, groups=groups, data_format="NCDHW")

    specific_op_input_layout(output, indices=0, layout="ND")
    specific_op_input_layout(output, indices=[1, 2], layout="NCDHW")
    specific_op_output_layout(output, indices=0, layout="NCDHW")
    return output


def conv3d_backward_bias_nocheck(grad: Tensor):
    return ge.ReduceSum(x=ge.FlattenV2(x=grad, axis=3), axes=[0, 2, 3], keep_dims=False)


def npu_conv3d_backward(x: Tensor, grad: Tensor, weight: Tensor, stride: List[int], padding: List[int],
                        dilation: List[int], groups: int, output_mask: List[bool]):
    grad_x = grad_weight = grad_bias = None
    if output_mask[0]:
        grad_x = conv3d_backward_input_nocheck(x, grad, weight, stride, padding, dilation, groups)
    if output_mask[1]:
        grad_weight = conv3d_backward_weight_nocheck(x, grad, weight, stride, padding, dilation, groups)
    if output_mask[2]:
        grad_bias = conv3d_backward_bias_nocheck(grad)
    return grad_x, grad_weight, grad_bias


def conv2d_backward_input_out_nocheck(x: Tensor, grad: Tensor, weight: Tensor, stride: List[int], padding: List[int],
                                      dilation: List[int], groups: int):
    strides = [1, 1, stride[0], stride[1]]
    pads = [padding[0], padding[0], padding[1], padding[1]]
    dilation = [1, 1, dilation[0], dilation[1]]
    if x._symsize is not None and all([not isinstance(s, torch.SymInt) for s in x._symsize]):
        input_size = x._symsize
    else:
        input_size = ge.Shape(x)
    output = ge.Conv2DBackpropInput(input_size=input_size, filter=weight, out_backprop=grad, strides=strides,
                                    pads=pads, dilations=dilation, groups=groups, data_format="NCHW")
    specific_op_input_layout(output, indices=0, layout="ND")
    specific_op_input_layout(output, indices=[1, 2], layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    return output


def conv2d_backward_weight_out_nocheck(x: Tensor, grad: Tensor, weight: Tensor, stride: List[int], padding: List[int],
                                       dilation: List[int], groups: int):
    strides = [1, 1, stride[0], stride[1]]
    pads = [padding[0], padding[0], padding[1], padding[1]]
    dilation = [1, 1, dilation[0], dilation[1]]
    output = ge.Conv2DBackpropFilter(x=x, filter_size=ge.Shape(weight), out_backprop=grad, strides=strides, pads=pads,
                                     dilations=dilation, groups=groups, data_format="NCHW")
    specific_op_input_layout(output, indices=1, layout="ND")
    specific_op_input_layout(output, indices=[0, 2], layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    return output


def conv2d_backward_bias_out_nocheck(grad: Tensor):
    return ge.ReduceSum(x=ge.FlattenV2(x=grad, axis=2), axes=[0, 2], keep_dims=False)


def npu_conv2d_backward(x: Tensor, grad: Tensor, weight: Tensor, stride: List[int], padding: List[int],
                        dilation: List[int], groups: int, output_mask: List[bool], input_is_3d: bool):
    grad_x = grad_weight = grad_bias = None
    if output_mask[0]:
        grad_x = conv2d_backward_input_out_nocheck(x, grad, weight, stride, padding, dilation, groups)
    if output_mask[1]:
        grad_weight = conv2d_backward_weight_out_nocheck(x, grad, weight, stride, padding, dilation, groups)
    if output_mask[2]:
        grad_bias = conv2d_backward_bias_out_nocheck(grad)
    if input_is_3d:
        if grad_x is not None:
            grad_x = ge.Squeeze(grad_x, axis=[2])
        if grad_weight is not None:
            grad_weight = ge.Squeeze(grad_weight, axis=[2])
    return grad_x, grad_weight, grad_bias


def conv_transpose2d_backward_input_out_nocheck(x, grad, weight, padding, output_padding, stride, dilation, groups):
    strides = [1, 1, stride[0], stride[1]]
    pads = [padding[0], padding[0], padding[1], padding[1]]
    dilation = [1, 1, dilation[0], dilation[1]]
    output = ge.Conv2D(x=grad, filter=weight, bias=None, offset_w=None, strides=strides, pads=pads,
                       dilations=dilation, groups=groups, data_format="NCHW")
    specific_op_input_layout(output, indices=[0, 1], layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    return output


def conv_transpose2d_backward_weight_out_nocheck(x, grad, weight, padding, output_padding, stride, dilation, groups):
    strides = [1, 1, stride[0], stride[1]]
    pads = [padding[0], padding[0], padding[1], padding[1]]
    dilation = [1, 1, dilation[0], dilation[1]]
    output = ge.Conv2DBackpropFilter(x=grad, filter_size=ge.Shape(weight), out_backprop=x, strides=strides, pads=pads,
                                     dilations=dilation, groups=groups, data_format="NCHW")
    specific_op_input_layout(output, indices=1, layout="ND")
    specific_op_input_layout(output, indices=[0, 2], layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    return output


def conv_transpose2d_backward_bias_out_nocheck(grad: Tensor):
    return ge.ReduceSum(x=ge.FlattenV2(x=grad, axis=2), axes=[0, 2], keep_dims=False)


def npu_conv_transpose2d_backward(x: Tensor, grad: Tensor, weight: Tensor, padding: List[int],
                                  output_padding: List[int], stride: List[int], dilation: List[int], groups: int,
                                  output_mask: List[bool], input_is_3d: bool):
    grad_x = grad_weight = grad_bias = None
    if output_mask[0]:
        grad_x = conv_transpose2d_backward_input_out_nocheck(x, grad, weight, padding, output_padding, stride, dilation,
                                                             groups)

    if output_mask[1]:
        grad_weight = conv_transpose2d_backward_weight_out_nocheck(x, grad, weight, padding,
                                                                   output_padding,
                                                                   stride, dilation, groups)
    if output_mask[2]:
        grad_bias = conv_transpose2d_backward_bias_out_nocheck(grad)

    if input_is_3d:
        if grad_x is not None:
            grad_x = ge.Squeeze(grad_x, axis=[2])
        if grad_weight is not None:
            grad_weight = ge.Squeeze(grad_weight, axis=[2])
    return grad_x, grad_weight, grad_bias


def conv_transpose3d_backward_input_out_nocheck(x, grad, weight, padding, output_padding, stride, dilation, groups):
    strides = [1, 1, stride[0], stride[1], stride[2]]
    pads = [padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]]
    dilation = [1, 1, dilation[0], dilation[1], dilation[2]]
    output = ge.Conv3D(x=grad, filter=weight, bias=None, offset_w=None, strides=strides, pads=pads,
                       dilations=dilation, groups=groups, data_format="NCDHW")
    specific_op_input_layout(output, indices=[0, 1], layout="NCDHW")
    specific_op_output_layout(output, indices=0, layout="NCDHW")
    return output


def conv_transpose3d_backward_bias_out_nocheck(grad):
    return ge.ReduceSum(x=ge.FlattenV2(x=grad, axis=3), axes=[0, 2, 3], keep_dims=False)


def conv_transpose3d_backward_weight_out_nocheck(x: Tensor, grad: Tensor, weight: Tensor, padding: List[int],
                                                 output_padding: List[int], stride: List[int], dilation: List[int],
                                                 groups: int):
    strides = [1, 1, stride[0], stride[1], stride[2]]
    pads = [padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]]
    dilation = [1, 1, dilation[0], dilation[1], dilation[2]]
    if weight._symsize is not None and all([not isinstance(s, torch.SymInt) for s in weight._symsize]):
        filter_size = weight._symsize
    else:
        filter_size = ge.Shape(weight)
    output = ge.Conv3DBackpropFilter(x=grad, out_backprop=x, filter_size=filter_size, strides=strides, pads=pads,
                                     dilations=dilation, groups=groups, data_format="NCDHW")
    specific_op_input_layout(output, indices=[0, 1, 2], layout="NCDHW")
    specific_op_output_layout(output, indices=0, layout="NCDHW")
    return output


def npu_conv_transpose3d_backward(x: Tensor, grad: Tensor, weight: Tensor, padding: List[int],
                                  output_padding: List[int], stride: List[int], dilation: List[int], groups: int,
                                  output_mask: List[bool]):
    grad_x = grad_weight = grad_bias = None
    if output_mask[0]:
        grad_x = conv_transpose3d_backward_input_out_nocheck(x, grad, weight, padding, output_padding, stride, dilation,
                                                             groups)

    if output_mask[1]:
        grad_weight = conv_transpose3d_backward_weight_out_nocheck(x, grad, weight, padding, output_padding, stride,
                                                                   dilation, groups)
    if output_mask[2]:
        grad_bias = conv_transpose3d_backward_bias_out_nocheck(grad)
    return grad_x, grad_weight, grad_bias


@declare_supported(
    [
        Support(F32(20, 33, 26, 100), F32(20, 16, 50, 100), F32(33, 16, 3, 5), bias_sizes=[33],
                stride=[2, 1], padding=[4, 2], dilation=[3, 1], transposed=False, output_padding=[0, 0], groups=1,
                output_mask=[True, True, True]),
        Support(F32(20, 33, 26, 100), F32(20, 16, 50, 100), F32(33, 16, 3, 5), bias_sizes=[33],
                stride=[2, 1], padding=[4, 2], dilation=[3, 1], transposed=True, output_padding=[0, 0], groups=1,
                output_mask=[True, True, True]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.convolution_backward.default)
def conveter_aten_convolution_backward_default(
        grad_output: Tensor,
        x: Tensor,
        weight: Tensor,
        bias_sizes: Optional[Union[List[int], Tensor]],
        stride: List[int],
        padding: Union[List[int], Tensor],
        dilation: List[int],
        transposed: bool,
        output_padding: Union[List[int], Tensor],
        groups: int,
        output_mask: List[bool],
        meta_outputs: TensorSpec = None,
):
    """NB: aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""
    if isinstance(padding, Tensor):
        raise NotImplementedError("torch.ops.aten.convolution.default ge converter is not implement "
                                  "when padding is tensor.")
    if isinstance(output_padding, Tensor):
        raise NotImplementedError("torch.ops.aten.convolution.default ge converter is not implement "
                                  "when output_padding is tensor.")
    grad, weight = dtype_promote(grad_output, weight, target_dtype=x.dtype)

    dim = x.rank
    input_is_3d = False
    if dim == 3:
        input_is_3d = True
        stride.insert(0, 1)
        padding.insert(0, 0)
        dilation.insert(0, 1)
        output_padding.insert(0, 0)
        x = ge.Unsqueeze(x, axes=[2])
        weight = ge.Unsqueeze(weight, axes=[2])
        grad = ge.Unsqueeze(grad, axes=[2])
    if not transposed:
        if dim == 4 or dim == 3:
            return npu_conv2d_backward(x, grad, weight, stride, padding, dilation, groups, output_mask, input_is_3d)
        elif dim == 5:
            logger.warning_once("conv3d only support non-generalized scenarios before 2024.02: "
                                "padding must be less than weight/filter/kernel."
                                "might be support generalized scenarios in future vision.")
            return npu_conv3d_backward(x, grad, weight, stride, padding, dilation, groups, output_mask)

    if dim == 4 or dim == 3:
        return npu_conv_transpose2d_backward(x, grad, weight, padding, output_padding, stride, dilation,
                                             groups, output_mask, input_is_3d)
    logger.warning_once("conv3d only support non-generalized scenarios before 2024.02: "
                        "padding must be less than weight/filter/kernel."
                        "might be support generalized scenarios in future vision.")
    return npu_conv_transpose3d_backward(x, grad, weight, padding, output_padding, stride, dilation, groups,
                                         output_mask)


@register_fx_node_ge_converter(torch.ops.aten.convolution_backward.out)
def conveter_aten_convolution_backward_out(
        grad_output: Tensor,
        x: Tensor,
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
        meta_outputs: TensorSpec = None
):
    """NB: aten::convolution_backward.out(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise AssertionError("torch.ops.aten.convolution_backward.out is redundant before pytorch 2.1.0, "
                         "might be supported in furture version.")
