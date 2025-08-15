import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_update_expert.default)
def convert_npu_moe_update_expert(
    expert_ids: Tensor,
    eplb_table: Tensor,
    *,
    expert_scales: Tensor = None,
    pruning_threshold: Tensor = None,
    active_mask: Tensor = None,
    local_rank_id: int = -1,
    world_size: int = -1,
    balance_mode: int = 0,
    meta_outputs: TensorSpec = None
):
    return ge.MoeUpdateExpert(
        expert_ids=expert_ids,
        eplb_table=eplb_table,
        expert_scales=expert_scales,
        pruning_threshold=pruning_threshold,
        active_mask=active_mask,
        local_rank_id=local_rank_id,
        world_size=world_size,
        balance_mode=balance_mode)