from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(4, 2048, 5120), F32(4, 2048, 5120), F32(5120,), F32(4, 2048, 1)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_rms_norm_backward.default)
def conveter_npu_rms_norm_backward_default(
    grad: Tensor,
    self: Tensor,
    gamma: Tensor,
    rstd: Tensor,
    meta_outputs: List[TensorSpec] = None
):
    """NB: npu::npu_rms_norm_backward(Tensor dy, Tensor self, Tensor gamma, Tensor rstd) -> (Tensor, Tensor)"""
    dx, dgamma = ge.RmsNormGrad(grad, self, rstd, gamma)
    return dx, dgamma
