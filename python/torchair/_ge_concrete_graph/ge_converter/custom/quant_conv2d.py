import torch
from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type, torch_dtype_value_to_ge_type, torch_dtype_value_to_ge_proto_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, BF16, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, Support

TORCH_DTYPE_MAP = {
    torch.float16: 5,
    torch.bfloat16: 15,
    torch.float32: 6,
    torch.float8_e5m2: 23,
    torch.float8_e4m3fn: 24,
    torch.bits8: 21,
    torch.int8: 1,
    torch.int32: 3,
}


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
    output_dtype: int = 0,
    bias: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    x_dtype: int = None,
    weight_dtype: int = None,
    meta_outputs: TensorSpec = None,
):
    """
    NB: aten::npu_quant_conv2d(Tensor input, Tensor weight, Tensor scale, int[2] strides=1,
        int[2] pads=0, int[2] dilations=1, int groups=1, int offset_x=0, str round_mode='rint',
        int output_dtype=0, Tensor? bias=None, Tensor? offset=None, int? x_dtype=None,
        int? weight_dtype=None) -> Tensor
    """
    if meta_outputs.rank != 4:
        raise NotImplementedError("torch.ops.npu.npu_quant_conv2d.default ge converter \
                                  is only implement for 4D tensor!")
    if offset is not None:
        raise NotImplementedError("torch.ops.npu.npu_quant_conv2d.default ge converter \
                                  is not implement while offset is not None!")
    import torch_npu
    valid_dtypes = {
        TORCH_DTYPE_MAP[torch.float16],
        TORCH_DTYPE_MAP[torch.float32],
        TORCH_DTYPE_MAP[torch.bfloat16],
        torch_npu.hifloat8
    }
    if output_dtype not in valid_dtypes:
        raise NotImplementedError("torch.ops.npu.npu_quant_conv2d.default ge converter \
                                  is not implement while output_dtype is not torch.float16 \
                                  or torch.bfloat16 or torch.float32 or torch_npu.hifloat8!")
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
    ge_output_dtype = torch_dtype_value_to_ge_proto_type(output_dtype)
    if x_dtype is not None:
        x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
    if weight_dtype is not None:
        weight.desc.dtype = torch_dtype_value_to_ge_proto_type(weight_dtype)

    out = ge.QuantConv2D(x, weight, scale, bias=bias, offset=None,
                         dtype=ge_output_dtype, strides=strides, pads=pads, dilations=dilations, groups=groups,
                         data_format="NCHW", offset_x=offset_x, round_mode=round_mode)
    if output_dtype == torch_npu.hifloat8:
        out.desc.dtype = torch_dtype_value_to_ge_proto_type(output_dtype)
    return out
