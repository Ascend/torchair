from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_dequant_swiglu_quant.default)
def conveter_npu_dequant_swiglu_quant_default(
        x: Tensor,
        weight_scale: Tensor = None,
        activation_scale: Tensor = None,
        bias: Tensor = None,
        quant_scale: Tensor = None,
        quant_offset: Tensor = None,
        group_index: Tensor = None,
        activate_left: bool = False,
        quant_mode: int = 0, 
        swiglu_mode: int = 0,         
        clamp_limit: float = 7.0,     
        glu_alpha: float = 1.702,     
        glu_bias: float = 1.0,        
        meta_outputs: TensorSpec = None):
    quant_mode_str = 'static'
    if quant_mode == 1:
        quant_mode_str = 'dynamic'

    return ge.DequantSwigluQuant(x, weight_scale, activation_scale, bias, quant_scale=quant_scale,
                                 quant_offset=quant_offset, group_index=group_index,
                                 activate_left=activate_left, quant_mode=quant_mode_str, 
                                 swiglu_mode=swiglu_mode, clamp_limit=clamp_limit, 
                                 glu_alpha=glu_alpha, glu_bias=glu_bias)
