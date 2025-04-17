from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt

from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, \
    specific_op_output_layout
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support
from torchair.core.utils import logger


def _cal_length(x, dilation, kernel_size, output_padding, padding, stride, index):
    x0 = ge.Mul(ge.Sub(x, 1), stride[index])
    x1 = ge.Mul(dilation[index], ge.Sub(kernel_size, 1))
    x2 = 1 - 2 * padding[index] + output_padding[index]
    x3 = ge.Add(x0, x1)
    return ge.Add(x3, x2)


def _npu_conv2d(x, weight, bias, stride, padding, dilation, groups):
    strides = [1, 1, stride[0], stride[1]]
    pads = [padding[0], padding[0], padding[1], padding[1]]
    dilations = [1, 1, dilation[0], dilation[1]]

    output = ge.Conv2D(x, weight, bias, None, strides=strides, pads=pads, dilations=dilations, groups=groups,
                       data_format="NCHW")
    specific_op_input_layout(output, indices=[0, 1, 2] if bias is not None else [0, 1], layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    return output


def _npu_conv3d(x, weight, bias, stride, padding, dilation, groups):
    strides = [1, 1, stride[0], stride[1], stride[2]]
    pads = [padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]]
    dilations = [1, 1, dilation[0], dilation[1], dilation[2]]

    output = ge.Conv3D(x, weight, bias, None, strides=strides, pads=pads, dilations=dilations, groups=groups,
                       data_format="NCDHW")
    if bias:
        specific_op_input_layout(output, indices=2, layout="ND")
    specific_op_input_layout(output, indices=[0, 1], layout="NCDHW")
    specific_op_output_layout(output, indices=0, layout="NCDHW")
    return output


def _npu_convolution(x, weight, bias, stride, padding, dilation, groups, meta_outputs):
    dim = meta_outputs.rank
    if dim == 4 or dim == 3:
        return _npu_conv2d(x, weight, bias, stride, padding, dilation, groups)
    else:
        logger.warning_once("conv3d only support non-generalized scenarios before 2024.02: "
                            "padding must be less than weight/filter/kernel."
                            "might be support generalized scenarios in future vision.")
        return _npu_conv3d(x, weight, bias, stride, padding, dilation, groups)


def _conv_transpose2d_npu_output_size(x, weight, padding, output_padding, stride, dilation, groups):
    input_shape = ge.Shape(x)
    n = ge.Gather(input_shape, 0)
    height_input = ge.Gather(input_shape, 2)
    weight_input = ge.Gather(input_shape, 3)
    weight_shape = ge.Shape(weight)
    c = ge.Mul(ge.Gather(weight_shape, 1), groups)
    kernel_size_2 = ge.Gather(weight_shape, 2)
    kernel_size_3 = ge.Gather(weight_shape, 3)
    h = _cal_length(height_input, dilation, kernel_size_2, output_padding, padding, stride, index=0)
    w = _cal_length(weight_input, dilation, kernel_size_3, output_padding, padding, stride, index=1)

    n, c, h, w = dtype_promote(n, c, h, w, target_dtype=DataType.DT_INT32)
    return ge.Pack([n, c, h, w], N=4, axis=0)


def _npu_conv_transpose2d(x, weight, bias, padding, output_padding, stride, dilation, groups):
    input_size = _conv_transpose2d_npu_output_size(x, weight, padding, output_padding, stride, dilation, groups)
    strides = [1, 1, stride[0], stride[1]]
    pads = [padding[0], padding[0], padding[1], padding[1]]
    dilation = [1, 1, dilation[0], dilation[1]]
    output_padding = [0, 0, output_padding[0], output_padding[1]]

    output = ge.Conv2DTranspose(input_size=input_size, x=x, filter=weight, bias=bias, offset_w=None,
                                strides=strides, pads=pads, dilations=dilation, groups=groups, data_format="NCHW",
                                output_padding=output_padding)
    specific_op_input_layout(output, indices=[0, 3] if bias is not None else [0], layout="ND")
    specific_op_input_layout(output, indices=[1, 2], layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    return output


def _conv_transpose3d_npu_output_size(x, w, padding, output_padding, stride, dilation, groups):
    input_shape = ge.Shape(x)
    n = ge.Gather(input_shape, 0)
    depth_input = ge.Gather(input_shape, 2)
    height_input = ge.Gather(input_shape, 3)
    weight_input = ge.Gather(input_shape, 4)

    weight_shape = ge.Shape(w)
    c = ge.Mul(ge.Gather(weight_shape, 1), groups)
    kernel_size_2 = ge.Gather(weight_shape, 2)
    kernel_size_3 = ge.Gather(weight_shape, 3)
    kernel_size_4 = ge.Gather(weight_shape, 4)

    d = _cal_length(depth_input, dilation, kernel_size_2, output_padding, padding, stride, index=0)
    h = _cal_length(height_input, dilation, kernel_size_3, output_padding, padding, stride, index=1)
    w = _cal_length(weight_input, dilation, kernel_size_4, output_padding, padding, stride, index=2)

    n, c, d, h, w = dtype_promote(n, c, d, h, w, target_dtype=DataType.DT_INT32)
    return ge.Pack([n, c, d, h, w], N=5, axis=0)


def _conv_transpose3d_npu_output_size_const(x: Tensor, weight: Tensor, padding: List[int], output_padding: List[int],
                                            stride: List[int], dilation: List[int], groups: int):
    input_shape = x._symsize
    n = input_shape[0]
    depth_input = input_shape[2]
    height_input = input_shape[3]
    weight_input = input_shape[4]

    weight_shape = weight._symsize
    c = weight_shape[1] * groups
    d = (depth_input - 1) * stride[0] - 2 * padding[0] + dilation[0] * (weight_shape[2] - 1) + output_padding[0] + 1
    h = (height_input - 1) * stride[1] - 2 * padding[1] + dilation[1] * (weight_shape[3] - 1) + output_padding[1] + 1
    w = (weight_input - 1) * stride[2] - 2 * padding[2] + dilation[2] * (weight_shape[4] - 1) + output_padding[2] + 1
    n, c, d, h, w = dtype_promote(n, c, d, h, w, target_dtype=DataType.DT_INT32)
    return ge.Pack([n, c, d, h, w], N=5, axis=0)


def _convolution_transpose3d_nocheck(x, weight, bias, padding, output_padding, stride, dilation, groups):
    if x._symsize is not None and all([not isinstance(s, torch.SymInt) for s in x._symsize]):
        input_size = _conv_transpose3d_npu_output_size_const(x, weight, padding, output_padding, stride, dilation,
                                                             groups)
    else:
        input_size = _conv_transpose3d_npu_output_size(x, weight, padding, output_padding, stride, dilation, groups)
    pads = [padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]]
    output_padding = [0, 0, 0, 0, 0]
    strides = [1, 1, stride[0], stride[1], stride[2]]
    dilation = [1, 1, dilation[0], dilation[1], dilation[2]]
    output = ge.Conv3DTranspose(input_size=input_size, x=x, filter=weight, bias=bias, offset_w=None,
                                strides=strides, pads=pads, dilations=dilation, groups=groups, data_format="NCDHW",
                                output_padding=output_padding)
    specific_op_input_layout(output, indices=[0, 3] if bias is not None else [0], layout="ND")

    specific_op_input_layout(output, indices=1, layout="NCDHW")
    specific_op_input_layout(output, indices=2, layout="NCDHW")
    specific_op_output_layout(output, indices=0, layout="NCDHW")
    return output


def _npu_convolution_transpose(x, weight, bias, padding, output_padding, stride, dilation, groups, meta_outputs):
    dim = meta_outputs.rank
    if dim == 4 or dim == 3:
        return _npu_conv_transpose2d(x, weight, bias, padding, output_padding, stride, dilation, groups)
    else:
        logger.warning_once("conv3d only support non-generalized scenarios before 2024.02: "
                            "padding must be less than weight/filter/kernel."
                            "might be support generalized scenarios in future vision.")
        return _convolution_transpose3d_nocheck(x, weight, bias, padding, output_padding, stride, dilation, groups)


@declare_supported(
    [
        Support(F32(4, 7, 4, 4), F32(4, 7, 3, 3), None, stride=[1, 1], padding=[1, 1], dilation=[1, 1],
                transposed=False, groups=1, output_padding=[0, 0]),
        Support(F32(4, 3, 4, 4), F32(3, 3, 1, 1), None, stride=[1, 1], padding=[0, 0], dilation=[1, 1],
                transposed=False, groups=1, output_padding=[0, 0]),
        Support(F32(4, 7, 2, 2), F32(4, 7, 3, 3), F32(4), stride=[2, 2], padding=[1, 1], dilation=[1, 1],
                transposed=False, groups=1, output_padding=[0, 0]),
        Support(F32(4, 7, 4, 4), F32(4, 7, 3, 3), None, stride=[1, 1], padding=[2, 2], dilation=[2, 2],
                transposed=False, groups=1, output_padding=[0, 0]),
        Support(F32(20, 16, 50, 100), F32(33, 16, 3, 5), F32(33, ), stride=[2, 1], padding=[4, 2],
                dilation=[3, 1], transposed=False, output_padding=[0, 0], groups=1),
        Support(F16(20, 16, 10, 50, 100), F32(33, 16, 3, 5, 2), F16(33, ), stride=[2, 1, 1],
                padding=[2, 2, 0], dilation=[1, 1, 1], transposed=False, output_padding=[0, 0, 0], groups=1),
        Support(F32(20, 16, 50, 100), F32(16, 33, 3, 5), F32(33, ), stride=[2, 1], padding=[4, 2],
                dilation=[1, 1], transposed=True, output_padding=[0, 0], groups=1),
        Support(F16(20, 16, 10, 50, 100), F16(16, 33, 3, 5, 2), F16(33, ), stride=[2, 1, 1],
                padding=[0, 1, 1], dilation=[1, 1, 1], transposed=True, output_padding=[0, 0, 0], groups=1),

    ]
)
@register_fx_node_ge_converter(torch.ops.aten.convolution.default)
def conveter_aten_convolution_default(
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        stride: List[int],
        padding: Union[List[int], Tensor],
        dilation: List[int],
        transposed: bool,
        output_padding: Union[List[int], Tensor],
        groups: int,
        meta_outputs: TensorSpec = None,
):
    """NB: aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups) -> Tensor"""
    if isinstance(padding, Tensor):
        raise NotImplementedError("torch.ops.aten.convolution.default ge converter is not implement "
                                  "when padding is tensor.")
    if isinstance(output_padding, Tensor):
        raise NotImplementedError("torch.ops.aten.convolution.default ge converter is not implement "
                                  "when output_padding is tensor.")
    if bias is not None:
        bias = dtype_promote(bias, target_dtype=meta_outputs.dtype)
    x, weight = dtype_promote(x, weight, target_dtype=meta_outputs.dtype)
    if len(padding) == 1:
        padding = padding * 3
    input_is_3d = False
    if meta_outputs.rank == 3:
        input_is_3d = True
        stride.insert(0, 1)
        padding.insert(0, 0)
        dilation.insert(0, 1)
        output_padding.insert(0, 0)
        x = ge.Unsqueeze(x, axes=[2])
        weight = ge.Unsqueeze(weight, axes=[2])
    if not transposed:
        output = _npu_convolution(x, weight, bias, stride, padding, dilation, groups, meta_outputs)
    else:
        output = _npu_convolution_transpose(x, weight, bias, padding, output_padding, stride,
                                            dilation, groups, meta_outputs)

    if input_is_3d:
        output = ge.Squeeze(output, axis=[2])
    return output


@register_fx_node_ge_converter(torch.ops.aten.convolution.out)
def conveter_aten_convolution_out(
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        stride: List[int],
        padding: Union[List[int], Tensor],
        dilation: List[int],
        transposed: bool,
        output_padding: Union[List[int], Tensor],
        groups: int,
        *,
        out: Tensor = None,
        meta_outputs: TensorSpec = None
):
    """NB: aten::convolution.out(Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, *, Tensor(a!) out) -> Tensor(a!)"""
    raise AssertionError(
        "torch.ops.aten.convolution.out is redundant before pytorch 2.1.0, might be supported in future version.")
