from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul_dequant.default)
def conveter_npu_quant_matmul_dequant_default(
    x: Tensor,
    quantized_weight: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    x_offset: Optional[Tensor] = None,
    smooth_scale: Optional[Tensor] = None,
    quant_mode: str = "pertoken",
    meta_outputs: TensorSpec = None
):
    """
    NB: aten::npu_quant_matmul_dequant(
        Tensor x, Tensor quantized_weight, Tensor weight_scale, *,
        Tensor? bias=None, Tensor? x_scale=None, Tensor? x_offset=None,
        Tensor? smooth_scale=None,
        str? quant_mode='pertoken') -> Tensor
    """
    return ge.QuantMatmulDequant(
        x, quantized_weight, weight_scale,
        bias=bias, x_scale=x_scale, x_offset=x_offset, smooth_scale=smooth_scale, x_quant_mode=quant_mode)