from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(1, 2, 3), F16(1, 2, 3), F16(3), output_mask=[True, True]),
    Support(F16(1, 2, 3), F16(1, 2, 3), F16(3), smooth_scale1=F16(3), output_mask=[True, True]),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_add_rms_norm_dynamic_quant.default)
def convert_npu_add_rms_norm_dynamic_quant_default(
    x1: Tensor,
    x2: Tensor,
    gamma: Tensor,
    smooth_scale1: Optional[Tensor] = None,
    smooth_scale2: Optional[Tensor] = None,
    beta: Optional[Tensor] = None,
    epsilon: float = 1e-6,
    output_mask: Optional[List] = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: npu::npu_add_rms_norm_dynamic_quant(Tensor x1, Tensor x2, Tensor gamma, *, Tensor? smooth_scale1=None, Tensor? smooth_scale2=None, Tensor? beta=None, float epsilon=1e-6, bool[2] output_mask=[]) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"""
    return ge.AddRmsNormDynamicQuant(x1, x2, gamma, smooth_scale1=smooth_scale1, smooth_scale2=smooth_scale2, beta=beta, epsilon=epsilon, output_mask=output_mask)