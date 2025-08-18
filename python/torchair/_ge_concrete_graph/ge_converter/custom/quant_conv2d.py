from torchair._ge_concrete_graph.ge_converter.converter_utils import *


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
