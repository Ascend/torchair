from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.core.utils import logger
from torchair.ge._ge_graph import DataType
from torchair._ge_concrete_graph.utils import convert_tensor_to_list, convert_tensor_to_dtype


def check_input_dtype(x, grad_output, weight):
    if x is not None and x.dtype == DataType.DT_FLOAT8_E4M3FN:
        raise TypeError("x dtype does not support 8-bit data types.")
    if grad_output is not None and grad_output.dtype == DataType.DT_FLOAT8_E4M3FN:
        raise TypeError("grad_output dtype does not support 8-bit data types.")
    if weight is not None and weight.dtype == DataType.DT_FLOAT8_E4M3FN:
        raise TypeError("weight dtype does not support 8-bit data types.")


def conv3d_backward_input_nocheck(x: Tensor, grad: Tensor, weight: Tensor, stride: List[int], padding: List[int],
                                  dilation: List[int], groups: int):
    strides = [1, 1, stride[0], stride[1], stride[2]]
    pads = [padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]]
    dilation = [1, 1, dilation[0], dilation[1], dilation[2]]

    check_input_dtype(x, grad, weight)

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
    check_input_dtype(x, grad, weight)
    output = ge.Conv3DBackpropFilter(x=x, filter_size=ge.Shape(weight), out_backprop=grad,
                                     strides=strides, pads=pads, dilations=dilation, groups=groups, data_format="NCDHW")

    specific_op_input_layout(output, indices=1, layout="ND")
    specific_op_input_layout(output, indices=[0, 2], layout="NCDHW")
    specific_op_output_layout(output, indices=0, layout="NCDHW")
    if x.dtype is not DataType.DT_FLOAT:
        output = ge.Cast(output, dst_type=x.dtype)
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
    check_input_dtype(None, grad, weight)
    output = ge.Conv2DBackpropInput(input_size=input_size, filter=weight, out_backprop=grad, strides=strides,
                                    pads=pads, dilations=dilation, groups=groups, data_format="NCHW")
    specific_op_input_layout(output, indices=0, layout="ND")
    specific_op_input_layout(output, indices=[1, 2], layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    return output


def conv2d_backward_weight_out_nocheck(x: Tensor, grad: Tensor, weight: Tensor, stride: List[int], padding: List[int],
                                       dilation: List[int], groups: int, target_dtype: int):
    strides = [1, 1, stride[0], stride[1]]
    pads = [padding[0], padding[0], padding[1], padding[1]]
    dilation = [1, 1, dilation[0], dilation[1]]
    check_input_dtype(x, grad, weight)
    output = ge.Conv2DBackpropFilter(x=x, filter_size=ge.Shape(weight), out_backprop=grad, strides=strides, pads=pads,
                                     dilations=dilation, groups=groups, data_format="NCHW")
    specific_op_input_layout(output, indices=1, layout="ND")
    specific_op_input_layout(output, indices=[0, 2], layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    if target_dtype is not DataType.DT_FLOAT:
        output = ge.Cast(output, dst_type=target_dtype)
    return output


def conv2d_backward_bias_out_nocheck(grad: Tensor):
    return ge.ReduceSum(x=ge.FlattenV2(x=grad, axis=2), axes=[0, 2], keep_dims=False)


def npu_conv2d_backward(x: Tensor, grad: Tensor, weight: Tensor, stride: List[int], padding: List[int],
                        dilation: List[int], groups: int, output_mask: List[bool], input_is_3d: bool, target_dtype: int):
    grad_x = grad_weight = grad_bias = None
    if output_mask[0]:
        grad_x = conv2d_backward_input_out_nocheck(x, grad, weight, stride, padding, dilation, groups)
    if output_mask[1]:
        grad_weight = conv2d_backward_weight_out_nocheck(x, grad, weight, stride, padding, dilation, groups, target_dtype)
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


def conv_transpose2d_backward_weight_out_nocheck(x, grad, weight, padding, output_padding, stride, dilation, groups, target_dtype):
    strides = [1, 1, stride[0], stride[1]]
    pads = [padding[0], padding[0], padding[1], padding[1]]
    dilation = [1, 1, dilation[0], dilation[1]]
    check_input_dtype(x, grad, weight)
    output = ge.Conv2DBackpropFilter(x=grad, filter_size=ge.Shape(weight), out_backprop=x, strides=strides, pads=pads,
                                     dilations=dilation, groups=groups, data_format="NCHW")
    specific_op_input_layout(output, indices=1, layout="ND")
    specific_op_input_layout(output, indices=[0, 2], layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    if target_dtype is not DataType.DT_FLOAT:
        output = ge.Cast(output, dst_type=target_dtype)
    return output


def conv_transpose2d_backward_bias_out_nocheck(grad: Tensor):
    return ge.ReduceSum(x=ge.FlattenV2(x=grad, axis=2), axes=[0, 2], keep_dims=False)


def npu_conv_transpose2d_backward(x: Tensor, grad: Tensor, weight: Tensor, padding: List[int],
                                  output_padding: List[int], stride: List[int], dilation: List[int], groups: int,
                                  output_mask: List[bool], input_is_3d: bool, target_dtype: int):
    grad_x = grad_weight = grad_bias = None
    if output_mask[0]:
        grad_x = conv_transpose2d_backward_input_out_nocheck(x, grad, weight, padding, output_padding, stride, dilation,
                                                             groups)

    if output_mask[1]:
        grad_weight = conv_transpose2d_backward_weight_out_nocheck(x, grad, weight, padding,
                                                                   output_padding,
                                                                   stride, dilation, groups, target_dtype)
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
    check_input_dtype(x, grad, weight)
    output = ge.Conv3DBackpropFilter(x=grad, out_backprop=x, filter_size=filter_size, strides=strides, pads=pads,
                                     dilations=dilation, groups=groups, data_format="NCDHW")
    specific_op_input_layout(output, indices=1, layout="ND")
    specific_op_input_layout(output, indices=[0, 2], layout="NCDHW")
    specific_op_output_layout(output, indices=0, layout="NCDHW")
    if grad.dtype is not DataType.DT_FLOAT:
        output = ge.Cast(output, dst_type=grad.dtype)
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
        logger.warning_once("torch.ops.aten.convolution_backward does not support dynamic graph scenarios where "
        "attributes [stride/dilation/padding/output_padding/groups] is of Tensor type, instead attributes is converted to List type.")
        padding = convert_tensor_to_list(padding, int)
    if isinstance(dilation, Tensor):
        dilation = convert_tensor_to_list(dilation, int)
    if isinstance(stride, Tensor):
        stride = convert_tensor_to_list(stride, int)
    if isinstance(output_padding, Tensor):
        output_padding = convert_tensor_to_list(output_padding, int)
    if isinstance(groups, Tensor):
        groups = convert_tensor_to_dtype(groups, int)
    if isinstance(bias_sizes, Tensor):
        bias_sizes = convert_tensor_to_list(bias_sizes, int)
    if isinstance(transposed, Tensor):
        transposed = convert_tensor_to_dtype(transposed, bool)
    if isinstance(output_mask, Tensor):
        output_mask = convert_tensor_to_list(output_mask, bool)

    grad, weight = dtype_promote(grad_output, weight, target_dtype=x.dtype)
    x_dtype = x.dtype
    grad_dtype = grad.dtype

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
            return npu_conv2d_backward(x, grad, weight, stride, padding, dilation, groups, output_mask, input_is_3d, x_dtype)
        elif dim == 5:
            return npu_conv3d_backward(x, grad, weight, stride, padding, dilation, groups, output_mask)

    if dim == 4 or dim == 3:
        return npu_conv_transpose2d_backward(x, grad, weight, padding, output_padding, stride, dilation,
                                             groups, output_mask, input_is_3d, grad_dtype)
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
