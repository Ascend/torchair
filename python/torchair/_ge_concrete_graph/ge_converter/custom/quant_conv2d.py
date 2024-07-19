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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported, \
    torch_type_to_ge_type
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import F32, I8, I32, Support


@declare_supported(
    [
        Support(I8(4, 7, 4, 4), I8(4, 7, 3, 3), F32(4), None, None, stride=[1, 1],
                padding=[1, 1], dilation=[1, 1], groups=1, offset_x=0),
        Support(I8(4, 3, 4, 4), I8(3, 3, 1, 1), F32(3), None, None, stride=[1, 1],
                padding=[0, 0], dilation=[1, 1], groups=1, offset_x=3),
        Support(I8(4, 7, 2, 2), I8(4, 7, 3, 3), F32(4), I32(4), None, stride=[2, 2],
                padding=[1, 1], dilation=[1, 1], groups=1, offset_x=0),
        Support(I8(4, 7, 4, 4), I8(4, 7, 3, 3), F32(4), None, None, stride=[1, 1],
                padding=[2, 2], dilation=[2, 2], groups=1, offset_x=5),
    ]
)


@register_fx_node_ge_converter(torch.ops.npu.npu_quant_conv2d.default)
def conveter_npu_npu_quant_conv2d(
    x: Tensor,
    weight: Tensor,
    scale: Tensor,
    stride: Optional[Union[List[int], Tensor]],
    pads: Optional[Union[List[int], Tensor]],
    dilations: Optional[Union[List[int], Tensor]],
    groups: int = 1,
    offset_x: int = 0,
    round_mode: str = 'rint',
    output_dtype: torch.dtype = None,
    bias: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None,
):
    """
    NB: aten::npu_quant_conv2d(Tensor input, Tensor weight, Tensor scale, int[2] strides=1,
        int[2] pads=0, int[2] dilations=1, int groups=1, int offset_x=0, str round_mode='rint',
        ScalarType? output_dtype=None, Tensor? bias=None, Tensor? offset=None) -> Tensor
    """
    if meta_outputs.rank != 4:
        raise NotImplementedError("torch.ops.npu.npu_quant_conv2d.default ge converter \
                                  is only implement for 4D tensor!")
    if offset is not None:
        raise NotImplementedError("torch.ops.npu.npu_quant_conv2d.default ge converter \
                                  is not implement while offset is not None!")
    if output_dtype != torch.float16:
        raise NotImplementedError("torch.ops.npu.npu_quant_conv2d.default ge converter \
                                  is not implement while dtype is not float16!")
    if stride is not None and isinstance(stride, Tensor):
        raise NotImplementedError("torch.ops.npu.npu_quant_conv2d.default ge converter \
                                  is not implement while stride is tensor!")
    if pads is not None and isinstance(pads, Tensor):
        raise NotImplementedError("torch.ops.npu.npu_quant_conv2d.default ge converter \
                                  is not implement while pad is tensor!")
    if dilations is not None and isinstance(dilations, Tensor):
        raise NotImplementedError("torch.ops.npu.npu_quant_conv2d.default ge converter \
                                  is not implement while dilation is tensor!")

    strides = [1, 1, stride[0], stride[1]]
    pads = [pads[0], pads[0], pads[1], pads[1]]
    dilations = [1, 1, dilations[0], dilations[1]]
    ge_output_dtype = torch_type_to_ge_type(output_dtype)

    return ge.QuantConv2D(x, weight, scale, bias=bias, offset=None,
                          dtype=ge_output_dtype, strides=strides, pads=pads, dilations=dilations, groups=groups,
                          data_format="NCHW", offset_x=offset_x, round_mode=round_mode)
