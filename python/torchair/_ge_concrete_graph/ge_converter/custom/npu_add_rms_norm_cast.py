from torchair._ge_concrete_graph.ge_converter.converter_utils import *

@register_fx_node_ge_converter(torch.ops.npu.npu_add_rms_norm_cast.default)
def convert_npu_add_rms_norm_cast_default(
    x1: Tensor,
    x2: Tensor,
    gamma: Tensor,
    epsilon: float = 1e-6,
    meta_outputs: List[TensorSpec] = None
):
    """NB: npu::npu_add_rms_norm_cast(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor, Tensor, Tensor)"""
    return ge.AddRmsNormCast(x1, x2, gamma, epsilon=epsilon)