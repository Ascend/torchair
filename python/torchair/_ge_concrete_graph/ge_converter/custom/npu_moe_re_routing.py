from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_re_routing.default)
def conveter_npu_moe_re_routing_default(
        tokens: Tensor,
        expert_token_num_per_rank: Tensor,
        *,
        per_token_scales: Optional[Tensor] = None,
        expert_token_num_type: int = 1,
        idx_type: int = 0,
        meta_outputs: List[TensorSpec] = None):
    return ge.MoeReRouting(tokens, expert_token_num_per_rank, per_token_scales,
                           expert_token_num_type=expert_token_num_type,
                           idx_type=idx_type)