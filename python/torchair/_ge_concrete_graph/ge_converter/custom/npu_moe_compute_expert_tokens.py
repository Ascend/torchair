from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_compute_expert_tokens.default)
def conveter_moe_compute_expert_tokens(
    sorted_experts: Tensor,
    num_experts: int,
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_moe_compute_expert_tokens(Tensor sorted_experts, int32 num_experts) -> Tensor"""
    return ge.MoeComputeExpertTokens(sorted_experts, num_experts=num_experts)
