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
        Support(F32(4, 7, 4, 4), F32(4, 7, 3, 3), None, stride=[1, 1], padding=[1, 1], dilation=[1, 1], transposed=False, groups=1, output_padding=[0, 0]),
        Support(F32(4, 3, 4, 4), F32(3, 3, 1, 1), None, stride=[1, 1], padding=[0, 0], dilation=[1, 1], transposed=False, groups=1, output_padding=[0, 0]),
        Support(F32(4, 7, 2, 2), F32(4, 7, 3, 3), F32(4), stride=[2, 2], padding=[1, 1], dilation=[1, 1], transposed=False, groups=1, output_padding=[0, 0]),
        Support(F32(4, 7, 4, 4), F32(4, 7, 3, 3), None, stride=[1, 1], padding=[2, 2], dilation=[2, 2], transposed=False, groups=1, output_padding=[0, 0]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.convolution.default)
def conveter_aten_convolution_default(
    input: Tensor,
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
    if meta_outputs.rank != 4:
        raise NotImplementedError("torch.ops.aten.convolution.default ge converter is only implement for 4D tensor input!")
    if transposed:
        """ output_padding is only used when transposed is True.""" 
        raise NotImplementedError("torch.ops.aten.convolution.default ge converter is not implement while transposed is True!")
    if isinstance(padding, Tensor):
        raise NotImplementedError("torch.ops.aten.convolution.default ge converter is not implement while padding is Tensor!")
    if groups > 1:
        raise NotImplementedError("torch.ops.aten.convolution.default ge converter is not implement while groups > 1!")
    if bias is not None:
        bias = dtype_promote(bias, target_dtype=meta_outputs.dtype)
    strides = [1, 1, stride[0], stride[1]]
    pads = [padding[0], padding[0], padding[1], padding[1]]
    dilations = [1, 1, dilation[0], dilation[1]]
    input, weight = dtype_promote(input, weight, target_dtype=meta_outputs.dtype)
    output = ge.Conv2D(input, weight, bias, None, strides=strides, pads=pads, dilations=dilations, groups=groups, data_format="NCHW")
    specific_op_input_layout(output, indices=[0, 1, 2] if bias is not None else [0, 1], layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    return output



@register_fx_node_ge_converter(torch.ops.aten.convolution.out)
def conveter_aten_convolution_out(
    input: Tensor,
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
    raise NotImplementedError("torch.ops.aten.convolution.out ge_converter is not implemented!")
