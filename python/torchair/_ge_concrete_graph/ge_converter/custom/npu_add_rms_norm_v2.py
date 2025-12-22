from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(1, 2, 3), F16(1, 2, 3), F16(3)),
    Support(F16(1, 2, 3), F16(1, 2, 3), F16(3), epsilon=1e-6),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_add_rms_norm_v2_functional.default)
def conveter_npu_add_rms_norm_v2_functional_default(
    x1: Tensor,
    x2: Tensor,
    gamma: Tensor,
    epsilon: float = 1e-6,
    meta_outputs: List[TensorSpec] = None
):
    """npu_add_rms_norm_v2_functional(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor, Tensor)"""
    x1_copy = ge.TensorMove(x1)
    x2_copy = ge.TensorMove(x2)
    
    out0, out1, out2 = ge.AddRmsNorm(x1_copy, x2_copy, gamma, epsilon=epsilon)
    return out1, out0, out2
