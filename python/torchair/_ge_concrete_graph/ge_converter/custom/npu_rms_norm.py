from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(8, 1, 128), F32(128)),
    Support(F16(8, 1, 128), F16(128)),
    # support setting epsilon
    Support(F16(8, 1, 128), F16(128), float(0.01)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_rms_norm.default)
def conveter_npu_rms_norm_default(
    self: Tensor,
    gamma: Tensor,
    epsilon: float = 1e-6,
    meta_outputs: List[TensorSpec] = None
):
    """NB: npu::npu_rms_norm(Tensor self, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor)"""
    return ge.RmsNorm(self, gamma, epsilon=epsilon)
