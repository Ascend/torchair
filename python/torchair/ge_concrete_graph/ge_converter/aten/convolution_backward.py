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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, \
    specific_op_output_layout
from torchair.ge_concrete_graph.supported_declaration import F32, F16, Support


@declare_supported(
    [
        Support(F32(4, 4, 4, 4), F32(4, 7, 4, 4), F32(4, 7, 3, 3), bias_sizes=[4], stride=[1, 1], padding=[1, 1], dilation=[1, 1], transposed=False, groups=1, output_padding=[0, 0], output_mask=[True, True, True]),
        Support(F32(4, 3, 4, 4), F32(4, 3, 4, 4), F32(3, 3, 1, 1), bias_sizes=None, stride=[1, 1], padding=[0, 0], dilation=[1, 1], transposed=False, groups=1, output_padding=[0, 0], output_mask=[True, True, False]),
        Support(F32(4, 4, 2, 2), F32(4, 7, 4, 4), F32(4, 7, 3, 3), bias_sizes=[4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], transposed=False, groups=1, output_padding=[0, 0], output_mask=[True, True, True]),
        Support(F32(4, 4, 4, 4), F32(4, 7, 4, 4), F32(4, 7, 3, 3), bias_sizes=None, stride=[1, 1], padding=[2, 2], dilation=[2, 2], transposed=False, groups=1, output_padding=[0, 0], output_mask=[True, True, False]),
    ]
)
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
    meta_outputs: TensorSpec = None,
):
    """NB: aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""
    if transposed:
        """ output_padding is only used when transposed is True.""" 
        raise NotImplementedError("torch.ops.aten.convolution_backward.default ge converter is not implement while transposed is True!")
    if isinstance(padding, Tensor):
        raise NotImplementedError("torch.ops.aten.convolution_backward.default ge converter is not implement while padding is Tensor!")
    if groups > 1:
        raise NotImplementedError("torch.ops.aten.convolution_backward.default ge converter is not implement while groups > 1!")
    strides = [1, 1, stride[0], stride[1]]
    pads = [padding[0], padding[0], padding[1], padding[1]]
    dilations = [1, 1, dilation[0], dilation[1]]
    grad_input = grad_weight = grad_bias = None
    if output_mask[0]:
        if meta_outputs[0].rank != 4:
            raise NotImplementedError("torch.ops.aten.convolution_backward.default ge converter is only implement for 4D tensor input!")
        input_size = ge.Shape(input)
        grad_output_cast0, weight_cast = dtype_promote(grad_output, weight, target_dtype=meta_outputs[0].dtype)
        grad_input = ge.Conv2DBackpropInput(input_size, weight_cast, grad_output_cast0, strides=strides, pads=pads, \
                                            dilations=dilations, groups=groups, data_format='NCHW')
        specific_op_input_layout(grad_input, indices=[1, 2], layout="NCHW")
        specific_op_output_layout(grad_input, indices=0, layout="NCHW")
    if output_mask[1]:
        if meta_outputs[1].rank != 4:
            raise NotImplementedError("torch.ops.aten.convolution_backward.default ge converter is only implement for 4D tensor weight!")
        weight_size = ge.Shape(weight)
        grad_output_cast1, input_cast = dtype_promote(grad_output, input, target_dtype=meta_outputs[1].dtype)
        grad_weight = ge.Conv2DBackpropFilter(input_cast, weight_size, grad_output_cast1, strides=strides, pads=pads, \
                                            dilations=dilations, groups=groups, data_format='NCHW')
        specific_op_input_layout(grad_weight, indices=[0, 2], layout="NCHW")
        specific_op_output_layout(grad_weight, indices=0, layout="NCHW")
    if output_mask[2]:
        grad_output = dtype_promote(grad_output, target_dtype=meta_outputs[2].dtype)
        grad_bias = ge.ReduceSum(grad_output, [i for i in range(meta_outputs[1].rank) if i != 1], keep_dims=False)
    return grad_input, grad_weight, grad_bias


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
    meta_outputs: TensorSpec = None
):
    """NB: aten::convolution_backward.out(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise NotImplementedError("torch.ops.aten.convolution_backward.out ge_converter is not implemented!")
