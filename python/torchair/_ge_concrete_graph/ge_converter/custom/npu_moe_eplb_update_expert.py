import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_eplb_update_expert.default)
def convert_npu_moe_eplb_update_expert(
    expert_ids: Tensor,
    eplb_table: Tensor,
    local_rank_id: int,
    world_size: int,
    *,
    balance_mode: int = 0,
    meta_outputs: TensorSpec = None
):
    return ge.MoeEplbUpdateExpert(
        expert_ids=expert_ids,
        eplb_table=eplb_table,
        local_rank_id=local_rank_id,
        world_size=world_size,
        balance_mode=balance_mode)