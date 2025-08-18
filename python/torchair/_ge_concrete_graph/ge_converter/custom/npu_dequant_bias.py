from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_dequant_bias.default)
def convert_npu_dequant_bias_default(
    x: Tensor,
    weight_scale: Tensor,
    activate_scale: Optional[Tensor],
    bias: Optional[Tensor],
    *,
    output_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_dequant_bias(Tensor x, Tensor weight_scale, Tensor? activation_scale, Tensor? bias, *,
                                 ScalarType? output_dtype=None) -> Tensor"""

    attr_output_type = DataType.DT_FLOAT16
    if output_dtype is not None:
        attr_output_type = torch_type_to_ge_type(output_dtype)
    
    return ge.dequant_bias(x, weight_scale, activate_scale, bias, output_dtype=attr_output_type)