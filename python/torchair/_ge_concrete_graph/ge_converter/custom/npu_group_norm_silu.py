from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(1, 32, 128, 64), F16(32), F16(32), group=32, eps=0.000100),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_group_norm_silu.default)
def conveter_npu_group_norm_silu_default(
    self: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    group: int,
    eps: float = 0.000100,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::native_group_norm_silu(Tensor self, Tensor? weight, Tensor? bias, int group,
                                        float eps) -> (Tensor, Tensor, Tensor)"""
    return ge.GroupNormSilu(self, weight, bias, num_groups=group, eps=eps)